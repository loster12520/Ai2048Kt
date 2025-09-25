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
 * 提供不同初始化策略的统一入口。
 * @see [doc/neural.md](../../doc/neural.md)
 */
interface Initialize {
    /**
     * 获取权重矩阵
     * @param fanIn 输入节点数
     * @param fanOut 输出节点数
     * @return 权重矩阵 $\mathbf{W}$
     * @see [doc/neural.md](../../doc/neural.md)
     */
    fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double>
    /**
     * 获取偏置向量
     * @param fanIn 输入节点数
     * @param fanOut 输出节点数
     * @return 偏置向量 $\mathbf{b}$
     * @see [doc/neural.md](../../doc/neural.md)
     */
    fun getBias(fanIn: Int, fanOut: Int): D1Array<Double>
}

/**
 * 全零初始化。
 * 所有参数初始化为0。
 * 优点：实现简单；缺点：无法打破对称性，网络无法有效训练。
 * $\mathbf{W}_{ij} = 0,\quad \mathbf{b}_j = 0$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class ZeroInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 0.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 0.0 }
}

/**
 * 全一初始化。
 * 所有参数初始化为1。
 * 优点：实现简单；缺点：同零初始化。
 * $\mathbf{W}_{ij} = 1,\quad \mathbf{b}_j = 1$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class OneInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 1.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 1.0 }
}

/**
 * 常数初始化。
 * 所有参数初始化为指定常���。
 * $\mathbf{W}_{ij} = c,\quad \mathbf{b}_j = c$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class ConstantInitialize(private val value: Double = 0.0) : Initialize {
    constructor(value: Number) : this(value.toDouble())

    override fun getWeight(fanIn: Int, fanOut: Int) = mk.d2array(fanIn, fanOut) { value }

    override fun getBias(fanIn: Int, fanOut: Int) = mk.d1array(fanOut) { value }
}

/**
 * 均匀分布初始化。
 * 权重和偏置从区间 $[min, max]$ 的均匀分布采样。
 * $\mathbf{W}_{ij} \sim \mathcal{U}(min, max)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class UniformInitialize(private val min: Double = 0.0, private val max: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(min, max) }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { Random.nextDouble(min, max) }
}

/**
 * 正态分布初始化。
 * 权重和偏置从均值 $mean$，标准差 $std$ 的正态分布采样。
 * $\mathbf{W}_{ij} \sim \mathcal{N}(mean, std^2)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class NormalInitialize(private val mean: Double = 0.0, private val std: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }
}

/**
 * Xavier 均匀分布初始化。
 * 适用于 sigmoid/tanh 激活函数。
 * $\mathbf{W}_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{fanIn + fanOut}}, \sqrt{\frac{6}{fanIn + fanOut}}\right)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class XavierUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }
}

/**
 * Xavier 正态分布初始化。
 * 适用于 sigmoid/tanh 激活函数。
 * $\mathbf{W}_{ij} \sim \mathcal{N}\left(0, \frac{2}{fanIn + fanOut}\right)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class XavierNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }
}

/**
 * He 均匀分布初始化。
 * 适用于 ReLU 激活函数。
 * $\mathbf{W}_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{fanIn}}, \sqrt{\frac{6}{fanIn}}\right)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class HeUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }
}

/**
 * He 正态分布初始化。
 * 适用于 ReLU 激活函数。
 * $\mathbf{W}_{ij} \sim \mathcal{N}\left(0, \frac{2}{fanIn}\right)$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class HeNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }
}