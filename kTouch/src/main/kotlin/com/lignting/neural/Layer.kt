package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.math.*
import kotlin.random.Random

/**
 * 神经网络层接口。
 *
 * 该接口定义了所有神经网络层的基本行为，包括前向传播（forward）和反向传播（backward）方法。
 * 任何实现该接口的类都可以作为神经网络的一层，支持参数更新与信息查询。
 *
 * 主要方法：
 * - forward：前向传播，计算输出。
 * - backward：反向传播，计算梯度并更新参数。
 * - copy：深拷贝当前层。
 * - info：返回层的简要信息。
 *
 * 常见实现类：
 * - Dense：全连接层，拥有可训练参数（权重和偏置），适用于大多数结构。
 * - Relu/Sigmoid/LeakyRelu/SoftPlus：激活层，无可训练参数，仅做非线性变换。
 * - Dropout：随机丢弃部分神经元，提升泛化能力。
 */
interface Layer {
    /**
     * 前向传播
     * @param input 输入数据
     * @return 输出数据
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
 *
 * 该层实现了 $y = xW + b$ 的线性变换，是最常见的神经网络层。
 *
 * 构造参数：
 * - inputSize：输入特征维度。
 * - outputSize：输出特征维度。
 * - weight：权重矩阵，形状为 (inputSize, outputSize)。
 * - bias：偏置向量，形状为 (outputSize)。
 *
 * 优点：
 * - 可学习参数，表达能力强。
 * - 适用于绝大多数结构。
 *
 * 缺点：
 * - 参数量大，易过拟合。
 * - 不适合处理空间结构数据（如图像）。
 *
 * 同类型对比：
 * - 卷积层（CNN）适合空间数据，参数更少。
 * - Dense层适合结构化数据和小型任务。
 *
 * 主要数学公式：
 * - $y = xW + b$
 *
 * @param inputSize 输入特征维度
 * @param outputSize 输出特征维度
 * @param weight 权重矩阵，形状为(inputSize, outputSize)
 * @param bias 偏置向量，形状为(outputSize)
 * @constructor 支持直接传入权重/偏置或通过初始化器生成
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
        val m = input.shape[0]
        val dWeight = (input.transpose() dot forwardOutput).times(1.0 / m)
        val dBias = mk.math.sumD2(forwardOutput, 0).times(1.0 / m)
        weight = optimizerCopy!!.optimizeW(weight, dWeight, scheduler, epoch)
        bias = optimizerCopy!!.optimizeB(bias, dBias, scheduler, epoch)
        return forwardOutput dot weight.transpose()
    }

    override fun copy(): Layer = Dense(inputSize, outputSize, weight.deepCopy(), bias.deepCopy())
    override fun info(): String =
        "Dense()\t\t\tinput:$inputSize\t\t\toutput:$outputSize\t\t\t"
}

/**
 * ReLU激活层。
 *
 * 该层实现 $y = \max(0, x)$ 的非线性变换，是最常用的激活函数之一。
 *
 * 构造参数：无。
 *
 * 优点：
 * - 计算高效，收敛快。
 * - 能有效缓解梯度消失。
 *
 * 缺点：
 * - 神经元可能“死亡”，即输出始终为0。
 *
 * 同类型对比：
 * - LeakyReLU 解决了死亡神经元问题。
 * - Sigmoid/SoftPlus 收敛慢但更平滑。
 *
 * 主要数学公式：
 * - $y = \max(0, x)$
 *
 * @constructor 无参数
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
 *
 * 该层实现 $y = \frac{1}{1 + e^{-x}}$ 的非线性变换，常用于二分类。
 *
 * 构造参数：
 * - zoom：缩放因子，控制输入的缩放，默认为1。
 *
 * 优点：
 * - 输出范围在(0,1)，适合概率建模。
 * - 理论成熟，易于理解。
 *
 * 缺点：
 * - 容易出现梯度消失。
 * - 收敛速度慢。
 *
 * 同类型对比：
 * - Relu收敛快但不适合概率输出。
 * - SoftPlus更平滑但计算复杂。
 *
 * 主要数学公式：
 * - $y = \frac{1}{1 + e^{-x}}$
 *
 * @param zoom 缩放因子，控制输入的缩放，默认为1
 * @constructor zoom: Int
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
 *
 * 该层实现 $y = \max(\alpha x, x)$ 的非线性变换，解决ReLU的“死亡神经元”问题。
 *
 * 构造参数：
 * - alpha：负区间斜率，默认为0.01。
 *
 * 优点：
 * - 保留负区间的微小梯度，避免神经元死亡。
 * - 收敛速度快。
 *
 * 缺点：
 * - 需手动设置alpha参数。
 *
 * 同类型对比：
 * - Relu简单但有死亡神经元。
 * - SoftPlus更平滑但计算复杂。
 *
 * 主要数学公式：
 * - $y = \max(\alpha x, x)$
 *
 * @param alpha 负区间斜率，默认为0.01
 * @constructor alpha: Double
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
 *
 * 该层实现 $y = \log(1 + e^x)$ 的平滑激活函数。
 *
 * 构造参数：
 * - base：对数底数，默认为2.0。
 * - maxClip：输入裁剪范围，防止溢出，默认为700.0。
 *
 * 优点：
 * - 平滑且连续，梯度稳定。
 * - 理论上可近似ReLU。
 *
 * 缺点：
 * - 计算复杂度高。
 * - 收敛速度慢于ReLU。
 *
 * 同类型对比：
 * - Relu收敛快但不平滑。
 * - Sigmoid适合概率输出。
 *
 * 主要数学公式：
 * - $y = \log(1 + e^x)$
 *
 * @param base 对数底数，默认为2.0
 * @param maxClip 输入裁剪范围，防止溢出，默认为700.0
 * @constructor base/maxClip: Double
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

class Softmax : Layer {
    override fun forward(input: D2Array<Double>): D2Array<Double> {
        val (batch, classes) = input.shape
        val result = mk.zeros<Double>(batch, classes)
        for (i in 0 until batch) {
            // 计算数值稳定的 softmax
            var maxVal = Double.NEGATIVE_INFINITY
            for (j in 0 until classes) {
                if (input[i, j] > maxVal) maxVal = input[i, j]
            }
            val exps = DoubleArray(classes) { j -> exp(input[i, j] - maxVal) }
            val sumExp = exps.sum()
            for (j in 0 until classes) {
                result[i, j] = exps[j] / sumExp
            }
        }
        return result
    }

    override fun backward(
        input: D2Array<Double>,
        forwardOutput: D2Array<Double>,
        optimizer: Optimizer,
        scheduler: Scheduler,
        epoch: Int
    ): D2Array<Double> {
        val probs = forward(input)
        val (batch, classes) = probs.shape
        val gradInput = mk.zeros<Double>(batch, classes)
        for (i in 0 until batch) {
            for (j in 0 until classes) {
                var sum = 0.0
                for (k in 0 until classes) {
                    val delta = if (j == k) 1 - probs[i, k] else -probs[i, k]
                    sum += forwardOutput[i, k] * probs[i, j] * delta
                }
                gradInput[i, j] = sum
            }
        }
        return gradInput
    }

    override fun copy() = Softmax()
    override fun info() = "Softmax()"
}

/**
 * Dropout层。
 *
 * 该层在训练时随机丢弃部分神经元，提升模型泛化能力。
 *
 * 构造参数：
 * - dropout：丢弃概率，默认为0.5。
 *
 * 优点：
 * - 有效防止过拟合。
 * - 实现简单，适用于各种网络。
 *
 * 缺点：
 * - 仅在训练时有效，推理时需关闭。
 * - 可能导致收敛变慢。
 *
 * 同类型对比：
 * - BatchNorm通过归一化提升泛化。
 * - Dropout通过随机丢弃提升泛化。
 *
 * @param dropout 丢弃概率，默认为0.5
 * @constructor dropout: Double
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