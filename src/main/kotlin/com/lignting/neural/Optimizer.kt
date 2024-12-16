package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.dnarray
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times

interface Optimizer<T, D : Dimension> {
    fun optimize(parameters: NDArray<T, D>, grads: NDArray<T, D>, learningRate: Double): NDArray<T, D>

    fun copy(): Optimizer<T, D>
}

//class GradientDescent<T : Double, D : Dimension> : Optimizer<T, D> {
//    override fun optimize(parameters: NDArray<T, D>, grads: NDArray<T, D>, learningRate: Double): NDArray<T, D> {
//        require(parameters.shape.contentEquals(grads.shape))
//        return parameters - (grads * mk.dnarray<Double, D>(parameters.shape) { learningRate })
//    }
//
//
//    override fun copy() = GradientDescent<T, D>()
//}