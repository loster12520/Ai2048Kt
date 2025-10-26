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
 * ### 优化器接口。
 *
 * 用于神经网络参数（权重和偏置）的优化更新，所有优化器需实现该接口。
 *
 * 主要方法：
 * - optimizeW：优化权重参数。
 * - optimizeB：优化偏置参数。
 * - copy：深拷贝当前优化器。
 *
 * 常见实现类：
 * - GradientDescent：标准梯度下降，简单高效。
 * - Momentum：带动量的梯度下降，收敛更快。
 * - Adam：结合Momentum和RMSProp，适合复杂场景。
 *
 * @param parameters 当前参数（权重或偏置）
 * @param grads 梯度
 * @param scheduler 学习率调度器
 * @param epoch 当前轮次
 * @return 更新后的参数
 */
interface Optimizer {
    /**
     * ### 优化权重
     * @param parameters 当前权重参数，形状为(inputSize, outputSize)
     * @param grads 权重梯度，形状为(inputSize, outputSize)
     * @param scheduler 学习率调度器
     * @param epoch 当前轮次
     * @return 更新后的权重参数
     */
    fun optimizeW(
        parameters: D2Array<Double>, grads: D2Array<Double>, scheduler: Scheduler, epoch: Int
    ): D2Array<Double>

    /**
     * ### 优化偏置
     * @param parameters 当前偏置参数，形状为(outputSize)
     * @param grads 偏置梯度，形状为(outputSize)
     * @param scheduler 学习率调度器
     * @param epoch 当前轮次
     * @return 更新后的偏置参数
     */
    fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>, scheduler: Scheduler, epoch: Int
    ): D1Array<Double>

    /**
     * ### 深拷贝当前优化器
     * @return 新的Optimizer实例
     */
    fun copy(): Optimizer
}

/**
 * ### 梯度下降优化器。
 *
 * 适用于大多数场景，公式简单。
 *
 * 优点：实现简单，收敛稳定。
 * 缺点：收敛速度慢，易陷入局部最优。
 *
 * 同类型对比：Momentum收敛更快，Adam适合复杂场景。
 *
 * 主要数学公式：
 * - $w = w - \eta \cdot \nabla w$
 *
 * @param parameters 当前参数
 * @param grads 梯度
 * @param scheduler 学习率调度器
 * @param epoch 当前轮次
 * @return 更新后的参数
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
 * ### Momentum优化器。
 *
 * 在梯度下降基础上引入动量项，加速收敛。
 *
 * 优点：收敛更快，能跳出局部最优。
 * 缺点：需调节beta参数。
 *
 * 同类型对比：GradientDescent实现简单，Adam更智能。
 *
 * 主要数学公式：
 * - $v_t = \beta v_{t-1} + (1-\beta)g_t$
 * - $w = w - \eta v_t$
 *
 * @param beta 动量系数，默认为0.9
 * @param parameters 当前参数
 * @param grads 梯度
 * @param scheduler 学习率调度器
 * @param epoch 当前轮次
 * @return 更新后的参数
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
 * ### Adam优化器。
 *
 * 结合Momentum和RMSProp，适合复杂场景。
 *
 * 优点：自适应学习率，收敛快，鲁棒性强。
 * 缺点：参数多，需调节。
 *
 * 同类型对比：GradientDescent实现简单，Momentum收敛快。
 *
 * 主要数学公式：
 * - $v_t = \beta_1 v_{t-1} + (1-\beta_1)g_t$
 * - $s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2$
 * - $w = w - \eta \frac{v_t}{\sqrt{s_t}+\epsilon}$
 *
 * @param beta1 一阶动量系数，默认为0.9
 * @param beta2 二阶动量系数，默认为0.999
 * @param epsilon 防止除零，默认为1e-5
 * @param maxGradBound 梯度裁剪阈值，默认为100.0
 * @param parameters 当前参数
 * @param grads 梯度
 * @param scheduler 学习率调度器
 * @param epoch 当前轮次
 * @return 更新后的参数
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

    // 使用滚动因子代替每次 pow 调用
    private var beta1PowW = 1.0
    private var beta2PowW = 1.0
    private var beta1PowB = 1.0
    private var beta2PowB = 1.0

    override fun optimizeW(
        parameters: D2Array<Double>, grads: D2Array<Double>, scheduler: Scheduler, epoch: Int
    ): D2Array<Double> {
        if (vWeight == null || sWeight == null) {
            vWeight = mk.zeros(parameters.shape, DataType.DoubleDataType)
            sWeight = mk.zeros(parameters.shape, DataType.DoubleDataType)
        }
        val lr = scheduler.getLearningRate(epoch)
        val clipped = grads.map { min(max(it, -maxGradBound), maxGradBound) }

        // 更新一阶和二阶矩（尽量合并分配）
        vWeight = vWeight!!.map { it * beta1 } + clipped.map { it * (1 - beta1) }
        sWeight = sWeight!!.map { it * beta2 } + (clipped * clipped).map { it * (1 - beta2) }

        // 更新滚动幂（避免 pow 调用）
        beta1PowW *= beta1
        beta2PowW *= beta2
        val vHat = vWeight!!.map { it / (1.0 - beta1PowW) }
        val sHat = sWeight!!.map { it / (1.0 - beta2PowW) }

        val denom = sHat.map { kotlin.math.sqrt(it) + epsilon }
        return parameters - (vHat / denom).map { it * lr }
    }

    override fun optimizeB(
        parameters: D1Array<Double>, grads: D1Array<Double>, scheduler: Scheduler, epoch: Int
    ): D1Array<Double> {
        if (vBias == null || sBias == null) {
            vBias = mk.zeros(parameters.shape, DataType.DoubleDataType)
            sBias = mk.zeros(parameters.shape, DataType.DoubleDataType)
        }

        val lr = scheduler.getLearningRate(epoch)
        val clipped = grads.map { min(max(it, -maxGradBound), maxGradBound) }

        vBias = vBias!!.map { it * beta1 } + clipped.map { it * (1 - beta1) }
        sBias = sBias!!.map { it * beta2 } + (clipped * clipped).map { it * (1 - beta2) }

        beta1PowB *= beta1
        beta2PowB *= beta2
        val vHat = vBias!!.map { it / (1.0 - beta1PowB) }
        val sHat = sBias!!.map { it / (1.0 - beta2PowB) }

        val denom = sHat.map { kotlin.math.sqrt(it) + epsilon }
        return parameters - (vHat / denom).map { it * lr }
    }

    override fun copy() = Adam(beta1, beta2, epsilon, maxGradBound)
}