package com.lignting.neural

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.reduce
import kotlin.math.pow

interface Loss {
    fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double
    fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double>
}

class Mse : Loss {
    override fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).reduce { a, i ->
            a + (i.pow(2))
        } / n
    }

    override fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double> {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).map { -it * 2 / n }
    }

}