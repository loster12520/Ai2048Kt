package com.lignting.neural

import kotlin.math.pow

/**
 * ### 学习率调度器接口。
 *
 * 用于动态调整优化器的学习率，所有调度器需实现该接口。
 *
 * 主要方法：
 * - getLearningRate：根据当前轮次返回学习率。
 *
 * 常见实现类：
 * - StepDecayScheduler：分步衰减，适合稳定训练。
 * - ExponentialScheduler：指数衰减，适合快速收敛。
 *
 * @param epoch 当前轮次
 * @return 学习率（Double）
 */
interface Scheduler {
    /**
     * ### 获取当前学习率
     * @param epoch 当前轮次
     * @return 学习率
     */
    fun getLearningRate(epoch: Int): Double
}

/**
 * ### 分步衰减调度器。
 *
 * 学习率随训练轮次线性衰减，适合稳定训练。
 *
 * 优点：实现简单，适合大多数场景。
 * 缺点：收敛速度可能较慢。
 *
 * 同类型对比：ExponentialScheduler收敛更快但可能不稳定。
 *
 * 主要数学公式：
 * - $\eta = \frac{\eta_0}{1 + epoch \cdot dropRate}$
 *
 * @param basicLearningRate 初始学习率
 * @param dropRate 衰减速率，默认为0.99
 * @param epoch 当前轮次
 * @return 当前学习率
 */
class StepDecayScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate / (1 + epoch * dropRate)
}

/**
 * ### 指数衰减调度器。
 *
 * 学习率随训练轮次指数衰减，适合快速收敛。
 *
 * 优点：收敛快，适合大规模训练。
 * 缺点：可能导致学习率过快降低。
 *
 * 同类型对比：StepDecayScheduler更稳定。
 *
 * 主要数学公式：
 * - $\eta = \eta_0 \cdot dropRate^{epoch}$
 *
 * @param basicLearningRate 初始学习率
 * @param dropRate 衰减速率，默认为0.99
 * @param epoch 当前轮次
 * @return 当前学习率
 */
class ExponentialScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate * (dropRate.pow(epoch))
}