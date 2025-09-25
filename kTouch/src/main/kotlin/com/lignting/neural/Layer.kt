package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.math.*
import kotlin.random.Random

/**
 * 神经网络层接口。
 * 包含前向传播和反向传播方法。
 * @see [doc/neural.md](../../doc/neural.md)
 */
interface Layer {
    /**
     * 前向传播
     * @param input 输入数据
     * @return 输出数据
     * @see [doc/neural.md](../../doc/neural.md)
     */
    fun forward(input: D2Array<Double>): D2Array<Double>
    /**
     * 反向传播
     * @param input 输入数据
     * @param forwardOutput 前向输出
     * @param optimizer 优化器
     * @param scheduler 学习率调度器
     * @param epoch 当前轮次
     * @return 梯度
     * @see [doc/neural.md](../../doc/neural.md)
     */
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

/**
 * 全连接层(Dense)。
 * $y = xW + b$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class Dense(
    private val inputSize: Int,
    private val outputSize: Int,
    private var weight: D2Array<Double>,
    private var bias: D1Array<Double>
) : Layer {

    var optimizerCopy: Optimizer? = null

    constructor(inputSize: Int, outputSize: Int, initialize: Initialize) : this(
        inputSize,
        outputSize,
        initialize.getWeight(inputSize, outputSize),
        initialize.getBias(inputSize, outputSize),
    )

    constructor(inputSize: Int, outputSize: Int) : this(
        inputSize, outputSize, UniformInitialize()
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

/**
 * ReLU激活层。
 * $y = \max(0, x)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class Relu() : Layer {
    override fun forward(input: D2Array<Double>): D2Array<Double> = input.map { max(it, 0.0) }

    override fun backward(
        input: D2Array<Double>, forwardOutput: D2Array<Double>, optimizer: Optimizer, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> = forwardOutput * input.map { if (it > 0) 1.0 else 0.0 }

    override fun copy() = Relu()
    override fun info(): String =
        "Relu()"
}

/**
 * Sigmoid激活层。
 * $y = \frac{1}{1 + e^{-x}}$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class Sigmoid(val zoom: Int = 1) : Layer {
    override fun forward(input: D2Array<Double>): D2Array<Double> = input.map { 1 / (1 + exp(-it / zoom)) }

    override fun backward(
        input: D2Array<Double>, forwardOutput: D2Array<Double>, optimizer: Optimizer, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> = forwardOutput * input.map { it * (1 - it) / zoom }

    override fun copy() = Sigmoid()
    override fun info(): String =
        "Sigmoid()"
}

/**
 * LeakyReLU激活层。
 * $y = \max(\alpha x, x)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class LeakyRelu(private val alpha: Double = 0.01) : Layer {
    override fun forward(input: D2Array<Double>): D2Array<Double> = input.map { if (it > 0) it else alpha * it }

    override fun backward(
        input: D2Array<Double>, forwardOutput: D2Array<Double>, optimizer: Optimizer, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> = forwardOutput * input.map { if (it > 0) 1.0 else alpha }

    override fun copy() = LeakyRelu(alpha)
    override fun info(): String =
        "LeakyRelu()"
}

/**
 * SoftPlus激活层。
 * $y = \log(1 + e^x)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class SoftPlus(private val base: Double = 2.0, private val maxClip: Double = 700.0) : Layer {
    override fun forward(input: D2Array<Double>) =
        input.map { log(1 + 1e-8 + base.pow(max(-maxClip, min(maxClip, it))), base = base) }

    override fun backward(
        input: D2Array<Double>,
        forwardOutput: D2Array<Double>,
        optimizer: Optimizer,
        scheduler: Scheduler,
        epoch: Int
    ) = forwardOutput * input.map {
        base.pow(-max(-maxClip, min(maxClip, it))) / (1 + base.pow(
            -max(
                -maxClip,
                min(maxClip, it)
            )
        ))
    }

    override fun copy() = SoftPlus(base, maxClip)

    override fun info() = "SoftPlus()"
}

/**
 * Dropout层。
 * 随机丢弃部分神经元。
 * @see [doc/neural.md](../../doc/neural.md)
 */
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