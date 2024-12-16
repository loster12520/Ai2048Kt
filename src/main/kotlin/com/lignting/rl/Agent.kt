package com.lignting.rl

import com.lignting.neural.Loss
import com.lignting.neural.Model
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.toMutableList
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import java.awt.Image
import javax.swing.JFrame
import kotlin.math.max
import kotlin.math.pow
import kotlin.random.Random

class Agent(val model: Model) {
    private var modelBuffer = model.copy()
    private var game = Game()
    private val replayBuffer = ReplayBuffer(10000)
    private val lossList = mutableListOf<Pair<Int, Double>>()
    private val gameList = mutableListOf<Pair<Int, Double>>()
    private val rewardList = mutableListOf<Pair<Int, Double>>()

    fun start(times: Int = 100000): Agent {
        var gameCount = 0
        var useStep = 0
        (1 until times).forEach {
            if (!game.isContinue()) {
                gameCount++
                if (gameCount % 10 == 0) {
                    println("$gameCount game failed at times $it")
                    game.print()
                    println("game use step: ${it - useStep}")
                    println("##############################################")
                }
                gameList.add(gameCount to game.score())
                game = Game()
                useStep = it
            }
            val score = game.reward()
            val panel = game.panel()
            val step = game.step()
            val directions = model.predict(mk.ndarray(panel.map { it.toDouble() }))
            val direction = directions.data.map { max(it, 0.01) }.mapIndexed { i, d -> i to d }
                .let {
                    if (Random.nextDouble() > 1 / (score + 1) * 100)
                        it.maxBy { it.second }.first
                    else
                        Random.nextInt(it.size)
                }
            game.move(direction)
//            game.print()
//            println(score)
            val reward = (game.reward() - score) + 100000.0 * (game.step() - step - 1)
            val maxNext = modelBuffer.predict(mk.ndarray(game.panel().map { it.toDouble() })).data.max()
            val update = directions.toMutableList()
            update[direction] = reward + 0.9 * maxNext
            replayBuffer.addReplay(panel, update)
            rewardList.add(it to reward)
            val trainData = replayBuffer.getTrainData(500).let {
                mk.ndarray(it.map { it.input.map { it.toDouble() } }) to mk.ndarray(it.map { it.output })
            }
            val loss = model.fit(trainData.first, trainData.second, it / 10000 + 1, 0.000001)
            lossList.add(it to loss)
//            if (it % 1000 == 0)
//                println("loss: $loss in $it times")

            if (it % 100 == 0) modelBuffer = model.copy()
        }
        return this
    }

    fun paintLoss(): Agent {
        val series = XYSeries("Loss")
        lossList.forEach {
            series.add(it.first, it.second)
        }
        val dataSet = XYSeriesCollection()
        dataSet.addSeries(series)
        val chart = ChartFactory.createXYLineChart(
            "Loss", // 图表标题
            "Times", // x轴标题
            "Loss", // y轴标题
            dataSet, // 数据集
            PlotOrientation.VERTICAL, // 垂直方向
            true, // 是否包含图例
            true, // 是否生成工具提示
            false // 是否生成URL链接
        )

        // 创建一个JFrame来显示图表
        val frame = JFrame()
        frame.defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
        frame.contentPane.add(ChartPanel(chart))
        frame.pack()
        frame.isVisible = true
        return this
    }

    fun paintGame(): Agent {
        val series = XYSeries("Game")
        gameList.forEach {
            series.add(it.first, it.second)
        }
        val dataSet = XYSeriesCollection()
        dataSet.addSeries(series)
        val chart = ChartFactory.createXYLineChart(
            "Game-Score", // 图表标题
            "Times", // x轴标题
            "Score", // y轴标题
            dataSet, // 数据集
            PlotOrientation.VERTICAL, // 垂直方向
            true, // 是否包含图例
            true, // 是否生成工具提示
            false // 是否生成URL链接
        )

        // 创建一个JFrame来显示图表
        val frame = JFrame()
        frame.defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
        frame.contentPane.add(ChartPanel(chart))
        frame.pack()
        frame.isVisible = true
        return this
    }

    fun paintReward(): Agent {
        val series = XYSeries("Game")
        rewardList.forEach {
            series.add(it.first, it.second)
        }
        val dataSet = XYSeriesCollection()
        dataSet.addSeries(series)
        val chart = ChartFactory.createXYLineChart(
            "Reward", // 图表标题
            "Times", // x轴标题
            "Reward", // y轴标题
            dataSet, // 数据集
            PlotOrientation.VERTICAL, // 垂直方向
            true, // 是否包含图例
            true, // 是否生成工具提示
            false // 是否生成URL链接
        )

        // 创建一个JFrame来显示图表
        val frame = JFrame()
        frame.defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
        frame.contentPane.add(ChartPanel(chart))
        frame.pack()
        frame.isVisible = true
        return this
    }

    fun printGame(): Agent {
        game.print()
        return this
    }
}