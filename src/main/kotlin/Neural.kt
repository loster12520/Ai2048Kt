import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import kotlin.math.exp
import kotlin.math.max

fun sigmoid(X: D2Array<Double>) = X.map { 1 / (1 + exp(-it)) }

fun relu(X: D2Array<Double>) = X.map { max(it / 100, it) }

fun sigmoid(x: Double) = 1 / (1 + exp(-x))

fun relu(x: Double) = 1 / (1 + exp(-x))

fun prev(A_prev: D2Array<Double>, W: D2Array<Double>, b: D1Array<Double>, activate: (Double) -> Double) =
    ((A_prev dot W) addB b).map(activate)

infix fun D2Array<Double>.addB(b: D1Array<Double>): D2Array<Double> {
    require(shape[0] == b.shape[0])
    return this + mk.ndarray(b.data.map { value ->
        (0..<shape[1]).map { value }
    }.flatten(), shape[0], shape[1])
}