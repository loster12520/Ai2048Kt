import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset

class Agent {
    val model = Sequential.of(
        Input(16),
        Dense(10, Activations.Relu),
        Dense(6, Activations.Relu),
        Dense(4)
    )

    var game = Game()

    fun start() {
        model.use {
            it.compile(
                optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
                loss = Losses.MSLE,
                metric = Metrics.MSLE
            )
        }
        (0..10000).forEach {

        }
    }

    fun step(model: Sequential) {
        if (!game.isContinue())
            game = Game()
        val nowState = game.panel()
        model.predict(nowState)
    }
}