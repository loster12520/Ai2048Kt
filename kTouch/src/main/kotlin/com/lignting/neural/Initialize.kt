package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.random.asJavaRandom


interface Initialize {
    fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double>
    fun getBias(fanIn: Int, fanOut: Int): D1Array<Double>
}

class ZeroInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 0.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 0.0 }
}

class OneInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> = mk.d2array(fanIn, fanOut) { 1.0 }
    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { 1.0 }
}

class ConstantInitialize(private val value: Double = 0.0) : Initialize {
    constructor(value: Number) : this(value.toDouble())

    override fun getWeight(fanIn: Int, fanOut: Int) = mk.d2array(fanIn, fanOut) { value }

    override fun getBias(fanIn: Int, fanOut: Int) = mk.d1array(fanOut) { value }
}

class UniformInitialize(private val min: Double = 0.0, private val max: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(min, max) }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> = mk.d1array(fanOut) { Random.nextDouble(min, max) }
}

class NormalInitialize(private val mean: Double = 0.0, private val std: Double = 1.0) : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int): D2Array<Double> =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }

    override fun getBias(fanIn: Int, fanOut: Int): D1Array<Double> =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * std + mean }
}

class XavierUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn + fanOut)), sqrt(6.0 / (fanIn + fanOut))) }
}

class XavierNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn + fanOut)) }
}

class HeUniformInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.nextDouble(-sqrt(6.0 / (fanIn)), sqrt(6.0 / (fanIn))) }
}

class HeNorMalInitialize : Initialize {
    override fun getWeight(fanIn: Int, fanOut: Int) =
        mk.d2array(fanIn, fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }

    override fun getBias(fanIn: Int, fanOut: Int) =
        mk.d1array(fanOut) { Random.asJavaRandom().nextGaussian() * sqrt(2.0 / (fanIn)) }
}