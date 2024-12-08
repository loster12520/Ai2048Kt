package com.lignting.rl

import com.lignting.neural.Model
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.toMutableList
import kotlin.math.max
import kotlin.math.pow
import kotlin.random.Random

class Agent(val model: Model) {
    var modelBuffer = model.copy()
    var game = Game()
    val replayBuffer = ReplayBuffer(10000)

    val modelList = mutableListOf<Model>()

    fun start(times: Int = 10000000) {
        (1 until times).forEach {
            if (!game.isContinue()) {
//                println("!!game failed in step ${game.step()}, with score ${game.score()}")
                game = Game()
            }
            val score = game.score()
            val panel = game.panel()
            val directions = modelBuffer.predict(mk.ndarray(panel.map { it.toDouble() }))
            val direction = directions.data.map { max(it, 0.01) }.mapIndexed { i, d -> i to d }
                .let {
                    if (Random.nextDouble() > 0.01)
                        it.maxBy { it.second }.first
                    else
                        Random.nextInt(it.size)
                }
            game.move(direction)
            val reward = (game.score() - score).pow(2) - 4
            val maxNext = modelBuffer.predict(mk.ndarray(game.panel().map { it.toDouble() })).data.max()
            val update = directions.toMutableList()
            update[direction] = reward + 0.9 * maxNext
            replayBuffer.addReplay(panel, update)
//            println(reward)
            val trainData = replayBuffer.getTrainData().let {
                mk.ndarray(it.map { it.input.map { it.toDouble() } }) to mk.ndarray(it.map { it.output })
            }
            val loss = model.fit(trainData.first, trainData.second)
            if (it % 1000 == 0)
                println("loss: $loss in $it times")

            if (it % 1000 == 0) modelBuffer = model.copy()
        }
    }
}