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

/**
 * 优化器接口，定义参数和梯度的更新方法。
 */
interface Optimizer {
    /**
     * 更新权重参数。
     */
    fun optimizeW(
        parameters: D2Array<Double>, grads: D2Array<Double>, scheduler: Scheduler, epoch: Int
    ): D2Array<Double>
    /**
     * 更新偏置参数。
     */
    fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>, scheduler: Scheduler, epoch: Int
    ): D1Array<Double>
    /**
     * 复制优化器实例。
     */
    fun copy(): Optimizer
}

/**
 * 梯度下降优化器（Gradient Descent）。
 * 公式：param = param - lr * grad
 */
class GradientDescent() : Optimizer {
    override fun optimizeW(
        parameters: D2Array<Double>, grads: D2Array<Double>, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> {
        require(parameters.shape contentEquals grads.shape)
        return parameters - grads.map { it * scheduler.getLearningRate(epoch) }
    }

    override fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>, scheduler: Scheduler, epoch: Int
    ): D1Array<Double> {
        require(parameters.shape contentEquals grads.shape)
        return parameters - grads.map { it * scheduler.getLearningRate(epoch) }
    }

    override fun copy() = GradientDescent()
}

/**
 * 动量优化器（Momentum）。
 * 公式：v = beta * v + (1-beta) * grad; param = param - lr * v
 * @param beta 动量因子，默认0.9
 */
class Momentum(val beta: Double = 0.9) : Optimizer {
    var vWeight: D2Array<Double>? = null
    var vBias: D1Array<Double>? = null
    override fun optimizeW(
        parameters: D2Array<Double>, grads: D2Array<Double>, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> {
        if (vWeight == null) vWeight = mk.zeros(parameters.shape, DataType.DoubleDataType)
        vWeight = vWeight!!.map { it * beta } + grads.map { it * (1 - beta) }
        return parameters - vWeight!!.map { it * scheduler.getLearningRate(epoch) }
    }

    override fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>, scheduler: Scheduler, epoch: Int
    ): D1Array<Double> {
        if (vBias == null) vBias = mk.zeros(parameters.shape, DataType.DoubleDataType)
        vBias = vBias!!.map { it * beta } + grads.map { it * (1 - beta) }
        return parameters - vBias!!.map { it * scheduler.getLearningRate(epoch) }
    }

    override fun copy() = Momentum(beta)
}

/**
 * Adam优化器。
 * 公式：
 * v = beta1 * v + (1-beta1) * grad
 * s = beta2 * s + (1-beta2) * grad^2
 * v_hat = v / (1 - beta1^t)
 * s_hat = s / (1 - beta2^t)
 * param = param - lr * v_hat / (sqrt(s_hat) + epsilon)
 * @param beta1 一阶矩动量因子，默认0.9
 * @param beta2 二阶矩动量因子，默认0.999
 * @param epsilon 防止除零，默认1e-5
 * @param maxGradBound 梯度裁剪阈值，默认100.0
 */
class Adam(
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
        parameters: D2Array<Double>, grads: D2Array<Double>, scheduler: Scheduler, epoch: Int
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
        return parameters - (vCorrected / sCorrected.map { it.pow(0.5) + epsilon }).map { it * scheduler.getLearningRate(epoch) }
    }

    override fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>, scheduler: Scheduler, epoch: Int
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
        return parameters - (vCorrected / sCorrected.map { it.pow(0.5) + epsilon }).map { it * scheduler.getLearningRate(epoch) }
    }

    override fun copy() = Adam(beta1, beta2, epsilon, maxGradBound)
}