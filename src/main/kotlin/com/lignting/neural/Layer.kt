package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.rand
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.math.*
import kotlin.random.Random

interface Layer {
    fun forward(input: D2Array<Double>): D2Array<Double>
    fun backward(
        input: D2Array<Double>,
        forwardOutput: D2Array<Double>,
        optimizer: Optimizer = GradientDescent(),
        scheduler: Scheduler,
        epoch: Int
    ): D2Array<Double>

    fun copy(): Layer
    fun info(): String
}

class Dense(
    private val inputSize: Int,
    private val outputSize: Int,
    private var weight: D2Array<Double>,
    private var bias: D1Array<Double>
) : Layer {

    var optimizerCopy: Optimizer? = null

    constructor(inputSize: Int, outputSize: Int) : this(
        inputSize, outputSize, mk.rand<Double>(inputSize, outputSize), mk.rand<Double>(outputSize)
    )

    override fun forward(input: D2Array<Double>): D2Array<Double> = (input dot weight) addB bias

    override fun backward(
        input: D2Array<Double>, forwardOutput: D2Array<Double>, optimizer: Optimizer, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> {
        if (optimizerCopy == null)
            optimizerCopy = optimizer
        val m = input.shape[1]
        val dWeight =
            (forwardOutput.transpose() dot input).map { 1.0 / m * it }
        val dBias = mk.math.sumD2(input, 0).map { 1.0 / m * it }
        weight = optimizerCopy!!.optimizeW(weight, dWeight, scheduler, epoch)
        bias = optimizerCopy!!.optimizeB(bias, dBias, scheduler, epoch)
        return input dot weight.transpose()
    }

    override fun copy(): Layer = Dense(inputSize, outputSize, weight.deepCopy(), bias.deepCopy())
    override fun info(): String =
        "Dense()\t\t\tinput:$inputSize\t\t\toutput:$outputSize\t\t\t"
}

class Relu() : Layer {
    override fun forward(input: D2Array<Double>): D2Array<Double> = input.map { max(it, 0.0) }

    override fun backward(
        input: D2Array<Double>, forwardOutput: D2Array<Double>, optimizer: Optimizer, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> = forwardOutput * input.map { if (it > 0) 1.0 else 0.0 }

    override fun copy() = Relu()
    override fun info(): String =
        "Relu()"
}

class Sigmoid(val zoom: Int = 1) : Layer {
    override fun forward(input: D2Array<Double>): D2Array<Double> = input.map { 1 / (1 + exp(-it / zoom)) }

    override fun backward(
        input: D2Array<Double>, forwardOutput: D2Array<Double>, optimizer: Optimizer, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> = forwardOutput * input.map { it * (1 - it) / zoom }

    override fun copy() = Sigmoid()
    override fun info(): String =
        "Sigmoid()"
}

class LeakyRelu(private val alpha: Double = 0.01) : Layer {
    override fun forward(input: D2Array<Double>): D2Array<Double> = input.map { if (it > 0) it else alpha * it }

    override fun backward(
        input: D2Array<Double>, forwardOutput: D2Array<Double>, optimizer: Optimizer, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> = forwardOutput * input.map { if (it > 0) 1.0 else alpha }

    override fun copy() = LeakyRelu(alpha)
    override fun info(): String =
        "LeakyRelu()"
}

class SoftPlus() : Layer {
    override fun forward(input: D2Array<Double>) = input.map { ln(1 + 1e-8 + exp(it)) }

    override fun backward(
        input: D2Array<Double>,
        forwardOutput: D2Array<Double>,
        optimizer: Optimizer,
        scheduler: Scheduler,
        epoch: Int
    ) = forwardOutput * input.map { 1 / (1 + exp(-it)) }

    override fun copy() = SoftPlus()

    override fun info() = "SoftPlus()"
}

class Dropout(private val dropout: Double = 0.5) : Layer {
    private var mask: D2Array<Double>? = null
    override fun forward(input: D2Array<Double>): D2Array<Double> {
        mask = mk.d2array(input.shape[0], input.shape[1]) { if (Random.nextDouble() > dropout) 1.0 else 0.0 }
        return (input * mask!!).map { it * (1.0 / (1.0 - dropout)) }
    }

    override fun backward(
        input: D2Array<Double>,
        forwardOutput: D2Array<Double>,
        optimizer: Optimizer,
        scheduler: Scheduler,
        epoch: Int
    ): D2Array<Double> =
        forwardOutput * (input * (mask
            ?: throw RuntimeException("backward before forward"))).map { it * (1.0 / (1.0 - dropout)) }

    override fun copy() = Dropout(dropout)

    override fun info() = "Dropout()\t\t\tdropout:$dropout"
}