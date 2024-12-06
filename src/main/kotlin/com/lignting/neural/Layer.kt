package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.rand
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus

interface Layer {
    fun forward(input: D2Array<Double>): D2Array<Double>
    fun backward(input: D2Array<Double>, forwardOutput: D2Array<Double>, learningRate: Double = 0.001)
}

class Dense(private val inputSize: Int, private val outputSize: Int) : Layer {
    private var weight = mk.rand<Double>(inputSize, outputSize)
    private var bias = mk.rand<Double>(outputSize)
    override fun forward(input: D2Array<Double>): D2Array<Double> =
        (input dot weight) addB bias

    override fun backward(input: D2Array<Double>, forwardOutput: D2Array<Double>, learningRate: Double) {
        val m = input.shape[1]
        val dWeight = (input dot forwardOutput.transpose()).map { 1.0 / m * it }
        val dBias = mk.math.sumD2(input, 1).map { 1.0 / m * it }
        weight -= dWeight
        bias -= dBias
    }
}