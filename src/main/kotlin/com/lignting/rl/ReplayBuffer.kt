package com.lignting.rl

import kotlin.random.Random


data class Replay(
    val input: List<Int>,
    val output: List<Double>,
    val tdError: Double = 0.0
)

class PriorityExperienceReplayBuffer(val size: Int = 1000) {
    private val replayList = mutableListOf<Replay>()

    fun addReplay(replay: Replay, canUpdate: Boolean = true) {
        if (replayList.size >= size) {
            replayList.removeAt(0)
            replayList.add(replay)
        } else {
            replayList.add(replay)
        }
        if (canUpdate) update()
    }

    fun update() = replayList.sortBy { it.tdError }

    fun addReplay(input: List<Int>, output: List<Double>, tdError: Double = 0.0, canUpdate: Boolean = true) =
        addReplay(Replay(input, output, tdError), canUpdate)

    fun addReplays(data: List<Pair<Pair<List<Int>, List<Double>>, Double>>) =
        data.forEach { (data, tdError) ->
            val (input, output) = data
            addReplay(input, output, tdError, false)
        }.let {
            update()
        }

    fun getTrainData(number: Int = (size / 20)): List<Replay> {
        val tdErrorSum = replayList.sumOf { it.tdError }
        val list = replayList.map { replay -> replay to replay.tdError / tdErrorSum }.reversed().toMutableList()
        val res = mutableListOf<Replay>()
        (1..number).forEach {
            var priority = Random.nextDouble()
            for ((replay, rate) in list) {
                if (priority > rate) priority -= rate
                else {
                    res.add(replay)
                    break
                }
            }
        }
        return res
    }
}