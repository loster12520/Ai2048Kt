package com.lignting.rl

import com.lignting.neural.*
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toMutableList
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import javax.swing.JFrame
import kotlin.math.max
import kotlin.random.Random

class Agent(val model: Model) {
    private var modelBuffer = model.copy()
    private val replayBuffer = PriorityExperienceReplayBuffer(100000)
    private val epsilonGetter = ExponentialScheduler(1.0, dropRate = 1 - 1e-1)
    private val lossList = mutableListOf<Double>()
    private val gameList = mutableListOf<Double>()
    private val maxAgentNumber = 100
    private val gamma = 0.99

    fun start(times: Int = 100): Agent {
        (1 until times).forEach {
            println("start $it epoches")
            epoches(it)
            println("-------------------------------------------------")
        }
        return this
    }

    private fun epoches(epoch: Int): MutableList<Game> {
        val (games, losses) = gameLoop(epoch = epoch)
        val score = games.map { it.score() }.average()
        val step = games.map { it.step() }.average()
        val loss = losses.average()
        println("games score: $score")
        println("games step: $step")
        println()
        val bestGame = games.maxBy { it.score() }
        println("best game:")
        bestGame.print()
        println()
        println("losses: $loss")
        lossList.addAll(losses)
        gameList.add(score)
        return games
    }

    private fun gameLoop(n: Int = maxAgentNumber, epoch: Int): Pair<MutableList<Game>, MutableList<Double>> {
        val gameList = MutableList(n) { Game() }
        val lossList = mutableListOf<Double>()
        var count = 0
        while (gameList.any { it.isContinue() }) {
            gameList.filter { it.isContinue() }.map { step(it, epoch) }.also { replayBuffer.addReplays(it) }
            lossList.add(train(epoch = epoch))
            count++
            if (count % 10 == 0) modelBuffer = model.copy()
        }
        modelBuffer = model.copy()
        return gameList to lossList
    }

    private fun step(game: Game, epoch: Int) = game.let {
        val panel = game.panel()
        val step = game.step()
        val directions = model.predictOne(mk.ndarray(panel.map { it.toDouble() }))
        val direction = directions.data.map { max(it, 0.01) }.mapIndexed { i, d -> i to d }
            .let {
                if (Random.nextDouble() > epsilonGetter.getLearningRate(epoch))
                    it.maxBy { it.second }.first
                else
                    Random.nextInt(it.size)
            }
        val reward = game.move(direction) + 100000.0 * (game.step() - step - 1)
        val maxNext = modelBuffer.predictOne(mk.ndarray(game.panel().map { it.toDouble() })).data.max()
        val update = directions.toMutableList()
        update[direction] = reward + gamma * maxNext

        val tdError = reward + 0.9 * maxNext - directions[direction]
        panel to update to tdError
    }

    private fun train(dataSize: Int = 100, epoch: Int): Double {
        val trainData = replayBuffer.getTrainData(dataSize).let {
            mk.ndarray(it.map { it.input.map { it.toDouble() } }) to mk.ndarray(it.map { it.output })
        }
        return model.fitWithBatchSize(trainData.first, trainData.second, epoch)
    }

    fun paintLoss(): Agent {
        val series = XYSeries("Loss")
        lossList.forEachIndexed() { index, value ->
            series.add(index, value)
        }
        val dataSet = XYSeriesCollection()
        dataSet.addSeries(series)
        val chart = ChartFactory.createXYLineChart(
            "Loss", // 图表标题
            "Epoches", // x轴标题
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
        gameList.forEachIndexed() { index, value ->
            series.add(index, value)
        }
        val dataSet = XYSeriesCollection()
        dataSet.addSeries(series)
        val chart = ChartFactory.createXYLineChart(
            "Game-Score", // 图表标题
            "Epoches", // x轴标题
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
}