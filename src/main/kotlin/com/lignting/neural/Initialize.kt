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
 * 参数初始化器接口，定义权重和偏置的初始化方法。
 */
interface Initialize {
    /**
     * 获取权重初始化值。
     */
    fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double>
    /**
     * 获取偏置初始化值。
     */
    fun getBias(fanIn: Int, fanOut: Int): D1Array<Double>
}

/**
 * 零初始化器，所有参数初始化为0。
 */
class ZeroInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 0.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 0.0 }
}

/**
 * 一初始化器，所有参数初始化为1。
 */
class OneInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 1.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 1.0 }
}

/**
 * 常数初始化器，所有参数初始化为指定常数。
 * @param value 初始化值
 */
class ConstantInitialize(private val value: Double = 0.0) : Initialize {
    constructor(value: Number) : this(value.toDouble())

    override fun getWeight(fanIn: Int, fanOut: Int) = mk.d2array(fanIn, fanOut) { value }

    override fun getBias(fanIn: Int, fanOut: Int) = mk.d1array(fanOut) { value }
}

/**
 * 均匀分布初始化器，参数在[min, max]区间均匀分布。
 */
class UniformInitialize(private val min: Double = 0.0, private val max: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(min, max) }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { Random.nextDouble(min, max) }
}

/**
 * 正态分布初始化器，参数服从N(mean, std^2)分布。
 */
class NormalInitialize(private val mean: Double = 0.0, private val std: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }
}

/**
 * Xavier均匀分布初始化器，适用于sigmoid/tanh激活。
 * 公式：U(-sqrt(6/(fanIn+fanOut)), sqrt(6/(fanIn+fanOut)))
 */
class XavierUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }
}

/**
 * Xavier正态分布初始化器，适用于sigmoid/tanh激活。
 * 公式：N(0, sqrt(2/(fanIn+fanOut)))
 */
class XavierNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }
}

/**
 * He均匀分布初始化器，适用于ReLU激活。
 * 公式：U(-sqrt(6/fanIn), sqrt(6/fanIn))
 */
class HeUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }
}

/**
 * He正态分布初始化器，适用于ReLU激活。
 * 公式：N(0, sqrt(2/fanIn))
 */
class HeNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }
}