package com.lignting.rl

import com.lignting.neural.Model
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.toMutableList
import kotlin.math.max
import kotlin.math.pow
import kotlin.random.Random

class Agent(val model: Model) {
    private var modelBuffer = model.copy()
    private var game = Game()
    private val replayBuffer = ReplayBuffer(100000)

    val modelList = mutableListOf<Model>()

    fun start(times: Int = 100000000) {
        var gameCount = 0
        var useStep = 0
        (1 until times).forEach {
            if (!game.isContinue()) {
                gameCount++
                if (gameCount % 10 == 0)
                    println("$gameCount game failed in step ${game.step()}, with score ${game.score()}")
                if (gameCount % 10 == 0) {
                    println("$gameCount game lass panel:")
                    game.print()
                }
                if (gameCount % 10 == 0) {
                    println("game use step: ${it - useStep}")
                }
                game = Game()
                useStep = it
            }
            val score = game.score()
            val panel = game.panel()
            val step = game.step()
            val directions = modelBuffer.predict(mk.ndarray(panel.map { it.toDouble() }))
            val direction = directions.data.map { max(it, 0.01) }.mapIndexed { i, d -> i to d }
                .let {
                    if (Random.nextDouble() > 0.01)
                        it.maxBy { it.second }.first
                    else
                        Random.nextInt(it.size)
                }
            game.move(direction)
            val reward = (game.score() - score).pow(2) - 2 - 10000 * (game.step() - step - 1)
            val maxNext = modelBuffer.predict(mk.ndarray(game.panel().map { it.toDouble() })).data.max()
            val update = directions.toMutableList()
            update[direction] = reward + 0.9 * maxNext
            replayBuffer.addReplay(panel, update)
            val trainData = replayBuffer.getTrainData(100).let {
                mk.ndarray(it.map { it.input.map { it.toDouble() } }) to mk.ndarray(it.map { it.output })
            }
            val loss = model.fit(trainData.first, trainData.second, it / 10000 + 1)
            if (it % 10000 == 0)
                println("loss: $loss in $it times")

            if (it % 100 == 0) modelBuffer = model.copy()
        }
    }
}