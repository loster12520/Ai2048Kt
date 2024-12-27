package com.lignting.rl

class ReplayBuffer(val size: Int = 1000) {
    data class Replay(
        val input: List<Int>,
        val output: List<Double>
    )

    private val replayList = mutableListOf<Replay>()

    fun addReplay(input: List<Int>, output: List<Double>) {
        if (replayList.size >= size) {
            replayList[randomIndex()] = Replay(input, output)
        } else {
            replayList.add(Replay(input, output))
        }
    }

    fun addReplays(data: List<Pair<List<Int>, List<Double>>>) =
        data.forEach { (input, output) -> addReplay(input, output) }

    fun getTrainData(number: Int = (size / 20)): List<Replay> {
        return ReservoirSample<Replay>(number).apply {
            replayList.forEach { add(it) }
        }.getResult()
    }

    private fun randomIndex(): Int {
        return (0 until replayList.size).random()
    }

    class ReservoirSample<T>(val k: Int) {
        private val samples = mutableListOf<T>()
        private val random = kotlin.random.Random.Default

        fun add(element: T) {
            if (samples.size < k) {
                samples.add(element)
            } else {
                val j = random.nextInt(samples.size + 1)
                if (j < k) {
                    samples[j] = element
                }
            }
        }

        fun getResult(): List<T> = samples
    }
}