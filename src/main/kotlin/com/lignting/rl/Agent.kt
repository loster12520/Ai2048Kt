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
import javax.swing.JFrame
import kotlin.math.max
import kotlin.random.Random

class Agent(val model: Model) {
    private var modelBuffer = model.copy()
    private val replayBuffer = ReplayBuffer(100000)
    private val lossList = mutableListOf<Double>()
    private val gameList = mutableListOf<Double>()
    private val maxAgentNumber = 100

    fun start(times: Int = 20): Agent {
        (1 until times).forEach {
            println("start $it epoches")
            epoches()
            println("-------------------------------------------------")
        }
        return this
    }

    private fun epoches(): MutableList<Game> {
        val (games, losses) = gameLoop()
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

    private fun gameLoop(n: Int = maxAgentNumber): Pair<MutableList<Game>, MutableList<Double>> {
        val gameList = MutableList(n) { Game() }
        val lossList = mutableListOf<Double>()
        var count = 0
        while (gameList.any { it.isContinue() }) {
            gameList.filter { it.isContinue() }.map { step(it) }.also { replayBuffer.addReplays(it) }
            lossList.add(train())
            count++
            if (count % 10 == 0) modelBuffer = model.copy()
        }
        modelBuffer = model.copy()
        return gameList to lossList
    }

    private fun step(game: Game) = game.let {
        val score = game.score()
        val panel = game.panel()
        val step = game.step()
        val directions = model.predictOne(mk.ndarray(panel.map { it.toDouble() }))
        val direction = directions.data.map { max(it, 0.01) }.mapIndexed { i, d -> i to d }
            .let {
                if (Random.nextDouble() > 1 / (score + 1) * 100)
                    it.maxBy { it.second }.first
                else
                    Random.nextInt(it.size)
            }
        game.move(direction)
        val reward = reward(panel, game.panel(), game) + 100000.0 * (game.step() - step - 1)
        val maxNext = modelBuffer.predictOne(mk.ndarray(game.panel().map { it.toDouble() })).data.max()
        val update = directions.toMutableList()
        update[direction] = reward + 0.9 * maxNext
        panel to update
    }

    private fun reward(last: List<Int>, now: List<Int>, game: Game): Double {
        // 基础奖励，可以根据需要调整
        val baseReward = 0.1

        // 检查得分是否增加
        val scoreIncrease = calculateScoreIncrease(last, now)

        // 检查是否形成了新数字
        val newTilesCreated = calculateNewTilesCreated(last, now)

        // 检查是否形成了2048
        val formed2048 = calculateFormed2048(now)

        // 检查游戏是否结束
        val gameEnded = !game.isContinue()

        // 计算奖励
        var totalReward = baseReward * scoreIncrease +
                baseReward * newTilesCreated -
                (if (gameEnded) 100.0 else 0.0) + // 游戏结束的惩罚
                100 * formed2048 // 形成2048的奖励

        // 如果游戏结束，给予额外的负奖励
        if (gameEnded) {
            totalReward -= 10.0
        }

        return totalReward
    }

    // 计算得分增加
    private fun calculateScoreIncrease(last: List<Int>, now: List<Int>) = now.sum() - last.sum()

    // 检查是否形成了新数字
    private fun calculateNewTilesCreated(last: List<Int>, now: List<Int>): Int {
        var newTiles = 0
        for (i in last.indices) {
            if (last[i] == 0 && now[i] != 0) {
                newTiles++
            }
        }
        return newTiles
    }

    // 检查是否形成了2048
    private fun calculateFormed2048(now: List<Int>) = if (now.contains(2048)) 1 else 0

    private fun train(dataSize: Int = 100): Double {
        val trainData = replayBuffer.getTrainData(dataSize).let {
            mk.ndarray(it.map { it.input.map { it.toDouble() } }) to mk.ndarray(it.map { it.output })
        }
        return model.fitWithBatchSize(trainData.first, trainData.second)
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