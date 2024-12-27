package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

interface Optimizer {
    fun optimizeW(parameters: D2Array<Double>, grads: D2Array<Double>): D2Array<Double>

    fun optimizeB(parameters: D1Array<Double>, grads: D1Array<Double>): D1Array<Double>

    fun copy(): Optimizer
}

class GradientDescent(val learningRate: Double = 0.01) : Optimizer {
    override fun optimizeW(parameters: D2Array<Double>, grads: D2Array<Double>): D2Array<Double> {
        require(parameters.shape contentEquals grads.shape)
        return parameters - grads.map { it * learningRate }
    }

    override fun optimizeB(parameters: D1Array<Double>, grads: D1Array<Double>): D1Array<Double> {
        require(parameters.shape contentEquals grads.shape)
        return parameters - grads.map { it * learningRate }
    }

    override fun copy() = GradientDescent(learningRate)
}

class Momentum(val learningRate: Double = 0.01, val beta: Double = 0.9) : Optimizer {
    var vWeight: D2Array<Double>? = null
    var vBias: D1Array<Double>? = null
    override fun optimizeW(
        parameters: D2Array<Double>, grads: D2Array<Double>
    ): D2Array<Double> {
        if (vWeight == null) vWeight = mk.zeros(parameters.shape, DataType.DoubleDataType)
        vWeight = vWeight!!.map { it * beta } + grads.map { it * (1 - beta) }
        return parameters - vWeight!!.map { it * learningRate }
    }

    override fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>
    ): D1Array<Double> {
        if (vBias == null) vBias = mk.zeros(parameters.shape, DataType.DoubleDataType)
        vBias = vBias!!.map { it * beta } + grads.map { it * (1 - beta) }
        return parameters - vBias!!.map { it * learningRate }
    }

    override fun copy() = Momentum(learningRate, beta)
}

class Adam(
    val learningRate: Double = 0.001,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1e-05,
    val maxGradBound: Double = 100.0
) :
    Optimizer {
    var vWeight: D2Array<Double>? = null
    var vBias: D1Array<Double>? = null
    var sWeight: D2Array<Double>? = null
    var sBias: D1Array<Double>? = null
    var wTimes: Int = 1
    var bTimes: Int = 1
    override fun optimizeW(
        parameters: D2Array<Double>, grads: D2Array<Double>
    ): D2Array<Double> {
        if (vWeight == null || sWeight == null) {
            vWeight = mk.zeros(parameters.shape, DataType.DoubleDataType)
            sWeight = mk.zeros(parameters.shape, DataType.DoubleDataType)
        }
        val clippedGrads = grads.map { min(max(it, -maxGradBound), maxGradBound) }
        vWeight = vWeight!!.map { it * beta1 } + clippedGrads.map { it * (1 - beta1) }
        val vCorrected = vWeight!!.map { it / (1 - beta1.pow(wTimes)) }
        sWeight = sWeight!!.map { it * beta2 } + (clippedGrads * clippedGrads).map { it * (1 - beta2) }
        val sCorrected = sWeight!!.map { it / (1 - beta2.pow(wTimes)) }
        wTimes++
        return parameters - (vCorrected / sCorrected.map { it.pow(0.5) + epsilon }).map { it * learningRate }
    }

    override fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>
    ): D1Array<Double> {
        if (vBias == null || sBias == null) {
            vBias = mk.zeros(parameters.shape, DataType.DoubleDataType)
            sBias = mk.zeros(parameters.shape, DataType.DoubleDataType)
        }
        val clippedGrads = grads.map { min(max(it, -maxGradBound), maxGradBound) }
        vBias = vBias!!.map { it * beta1 } + clippedGrads.map { it * (1 - beta1) }
        val vCorrected = vBias!!.map { it / (1 - beta1.pow(bTimes)) }
        sBias = sBias!!.map { it * beta2 } + (clippedGrads * clippedGrads).map { it * (1 - beta2) }
        val sCorrected = sBias!!.map { it / (1 - beta2.pow(bTimes)) }
        bTimes++
        return parameters - (vCorrected / sCorrected.map { it.pow(0.5) + epsilon }).map { it * learningRate }
    }

    override fun copy() = Adam(learningRate, beta1, beta2, epsilon, maxGradBound)
}