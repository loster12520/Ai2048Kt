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
 * 神经网络层接口，定义了前向传播、反向传播、复制和信息获取方法。
 */
interface Layer {
    /**
     * 前向传播，计算输出。
     * @param input 输入数据
     * @return 输出数据
     */
    fun forward(input: D2Array<Double>): D2Array<Double>
    /**
     * 反向传播，计算梯度并更新参数。
     * @param input 输入数据
     * @param forwardOutput 前向输出
     * @param optimizer 优化器
     * @param scheduler 学习率调度器
     * @param epoch 当前轮次
     * @return 梯度
     */
    fun backward(
        input: D2Array<Double>,
        forwardOutput: D2Array<Double>,
        optimizer: Optimizer = GradientDescent(),
        scheduler: Scheduler,
        epoch: Int
    ): D2Array<Double>
    /**
     * 复制当前层。
     */
    fun copy(): Layer
    /**
     * 获取层信息字符串。
     */
    fun info(): String
}

/**
 * 全连接层（Dense Layer），实现线性变换：output = input · weight + bias。
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
 * ReLU激活层，公式：output = max(0, input)
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
 * Sigmoid激活层，公式：output = 1 / (1 + exp(-input / zoom))
 * @param zoom 缩放因子，默认为1
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
 * LeakyReLU激活层，公式：output = input (input>0), alpha*input (input<=0)
 * @param alpha 负区间斜率，默认为0.01
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