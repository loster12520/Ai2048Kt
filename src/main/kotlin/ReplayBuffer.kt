import java.util.*

class ReplayBuffer(val size: Int = 1000) {
    data class Replay(
        val input: List<Int>,
        val output: List<Double>
    )

    val replayList = mutableListOf<Replay>()

    fun addReplay(input: List<Int>, output: List<Double>) =
        replayList.add(Replay(input, output)).also { if (replayList.size > size) replayList.removeFirst() }

    fun getTrainData(number: Int = (size / 5)) = replayList.shuffled().take(number).toList()
}