package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.random.asJavaRandom

/**
 * 权重和偏置初始化接口。
 *
 * 提供不同初始化策略的统一入口。
 */
interface Initialize {
    /**
     * 获取权重矩阵
     *
     * @param fanIn 输入节点数
     * @param fanOut 输出节点数
     * @return 权重矩阵 $\mathbf{W}$
     */
    fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double>
    /**
     * 获取偏置向量
     * @param fanIn 输入节点数
     * @param fanOut 输出节点数
     * @return 偏置向量 $\mathbf{b}$
     */
    fun getBias(fanIn: Int, fanOut: Int): D1Array<Double>
}

/**
 * 全零初始化。
 *
 * 所有参数初始化为0。
 *
 * 优点：实现简单；缺点：无法打破对称性，网络无法有效训练。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} = 0,\quad \mathbf{b}_j = 0$
 */
class ZeroInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 0.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 0.0 }
}

/**
 * 全一初始化。
 *
 * 所有参数初始化为1。
 *
 * 优点：实现简单；缺点：同零初始化。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} = 1,\quad \mathbf{b}_j = 1$
 */
class OneInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 1.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 1.0 }
}

/**
 * 常数初始化。
 *
 * 所有参数初始化为指定常量。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} = c,\quad \mathbf{b}_j = c$
 */
class ConstantInitialize(private val value: Double = 0.0) : Initialize {
    constructor(value: Number) : this(value.toDouble())

    override fun getWeight(fanIn: Int, fanOut: Int) = mk.d2array(fanIn, fanOut) { value }

    override fun getBias(fanIn: Int, fanOut: Int) = mk.d1array(fanOut) { value }
}

/**
 * 朴素的均匀分布初始化。
 *
 * 这种方法是早期最直接的想法：从一个固定的、简单的分布中随机采样。
 *
 * 优点：
 * - **实现简单**：概念直观，代码实现容易。
 * - **打破了对称性**：由于是随机初始化，满足了最基本的要求。
 *
 * 缺点：
 * - **严重依赖经验**：`std` 或范围的选择非常关键，且没有一个普适的理论指导。如果设置得太小，会导致梯度消失；如果设置得太大，会导致梯度爆炸。
 * - **不适用于深层网络**：随着网络层数的增加，激活输出和梯度的方差会急剧缩小或放大，使得深层网络极难训练。
 * - **对不同激活函数不敏感**：它没有考虑后续使用的激活函数（如 Sigmoid, Tanh, ReLU）的特性。
 *
 * > 结论：这种方法在现代深度学习模型中基本已被淘汰，仅用于教学或非常浅层的网络。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} \sim \mathcal{U}(min, max)$
 */
class UniformInitialize(private val min: Double = 0.0, private val max: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(min, max) }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { Random.nextDouble(min, max) }
}

/**
 * 朴素的正态分布初始化。
 *
 * 这种方法是早期最直接的想法：从一个固定的、简单的分布中随机采样。
 *
 * 优点：
 * - **实现简单**：概念直观，代码实现容易。
 * - **打破了对称性**：由于是随机初始化，满足了最基本的要求。
 *
 * 缺点：
 * - **严重依赖经验**：`std` 或范围的选择非常关键，且没有一个普适的理论指导。如果设置得太小，会导致梯度消失；如果设置得太大，会导致梯度爆炸。
 * - **不适用于深层网络**：随着网络层数的增加，激活输出和梯度的方差会急剧缩小或放大，使得深层网络极难训练。
 * - **对不同激活函数不敏感**：它没有考虑后续使用的激活函数（如 Sigmoid, Tanh, ReLU）的特性。
 *
 * > 结论：这种方法在现代深度学习模型中基本已被淘汰，仅用于教学或非常浅层的网络。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} \sim \mathcal{N}(mean, std^2)$
 */
class NormalInitialize(private val mean: Double = 0.0, private val std: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }
}

/**
 * Xavier/Glorot 均匀分布初始化。
 *
 * Xavier 初始化是解决朴素初始化问题的第一个重要突破。它明确考虑了饱和型激活函数（如 Tanh, Sigmoid），其目标是保持正向传播时，每一层输出的方差等于其输入的方差。
 *
 * 优点：
 * - **理论驱动**：有坚实的数学推导，确保了信号在前向传播中的稳定性。
 * - **显著改善收敛**：对于 Tanh 和 Sigmoid 等激活函数，能有效防止梯度消失/爆炸，大大加快了收敛速度。
 * - **成为饱和激活函数的标准**：在 ReLU 流行之前，这是最先进且最常用的初始化方法。
 *
 * 缺点：
 * - **对 ReLU 家族效果不佳**：ReLU 激活函数会将一半的神经元输出置零，这破坏了 Xavier 初始化所依赖的方差保持假设。使用 Xavier 初始化 ReLU 网络时，方差仍然会逐层衰减。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{fanIn + fanOut}}, \sqrt{\frac{6}{fanIn + fanOut}}\right)$
 */
class XavierUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }
}

/**
 * Xavier/Glorot 正态分布初始化。
 *
 * Xavier 初始化是解决朴素初始化问题的第一个重要突破。它明确考虑了饱和型激活函数（如 Tanh, Sigmoid），其目标是保持正向传播时，每一层输出的方差等于其输入的方差。
 *
 * 优点：
 * - **理论驱动**：有坚实的数学推导，确保了信号在前向传播中的稳定性。
 * - **显著改善收敛**：对于 Tanh 和 Sigmoid 等激活函数，能有效防止梯度消失/爆炸，大大加快了收敛速度。
 * - **成为饱和激活函数的标准**：在 ReLU 流行之前，这是最先进且最常用的初始化方法。
 *
 * 缺点：
 * - **对 ReLU 家族效果不佳**：ReLU 激活函数会将一半的神经元输出置零，这破坏了 Xavier 初始化所依赖的方差保持假设。使用 Xavier 初始化 ReLU 网络时，方差仍然会逐层衰减。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} \sim \mathcal{N}\left(0, \frac{2}{fanIn + fanOut}\right)$
 */
class XavierNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }
}

/**
 * He(MSRA) 均匀分布初始化。
 *
 * He 初始化是专门为 ReLU 及其变体（如 Leaky ReLU, PReLU） 设计的。它修正了 Xavier 的假设，只考虑正向传播时一半神经元被激活的情况，因此将方差放大了一倍。
 *
 * 优点：
 * - **ReLU 网络的黄金标准**：对于使用 ReLU 的网络（如大多数 CNN 和现代全连接网络），He 初始化是目前最有效、最常用的方法。
 * - **有效解决深度网络训练难题**：使得训练极深的网络（如 ResNet）成为可能。
 * - **收敛更快更稳定**：相比 Xavier，它能带来更快的收敛速度和更高的最终精度。
 *
 * 缺点：
 * - **不适用于饱和激活函数**：如果错误地将其用于 Tanh 或 Sigmoid 网络，可能会导致梯度爆炸，因为初始权重过大。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{fanIn}}, \sqrt{\frac{6}{fanIn}}\right)$
 */
class HeUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }
}

/**
 * He(MSRA) 正态分布初始化。
 *
 * He 初始化是专门为 ReLU 及其变体（如 Leaky ReLU, PReLU） 设计的。它修正了 Xavier 的假设，只考虑正向传播时一半神经元被激活的情况，因此将方差放大了一倍。
 *
 * 优点：
 * - **ReLU 网络的黄金标准**：对于使用 ReLU 的网络（如大多数 CNN 和现代全连接网络），He 初始化是目前最有效、最常用的方法。
 * - **有效解决深度网络训练难题**：使得训练极深的网络（如 ResNet）成为可能。
 * - **收敛更快更稳定**：相比 Xavier，它能带来更快的收敛速度和更高的最终精度。
 *
 * 缺点：
 * - **不适用于饱和激活函数**：如果错误地将其用于 Tanh 或 Sigmoid 网络，可能会导致梯度爆炸，因为初始权重过大。
 *
 * 主要数学公式：
 * - $\mathbf{W}_{ij} \sim \mathcal{N}\left(0, \frac{2}{fanIn}\right)$
 */
class HeNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }
}