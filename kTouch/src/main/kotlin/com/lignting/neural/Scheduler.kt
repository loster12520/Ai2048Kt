package com.lignting.neural

import kotlin.math.pow

/**
 * 学习率调度器接口。
 * @see [doc/neural.md](../../doc/neural.md)
 */
interface Scheduler {
    /**
     * 获取当前学习率
     * @param epoch 当前轮次
     * @return 学习率
     * @see [doc/neural.md](../../doc/neural.md)
     */
    fun getLearningRate(epoch: Int): Double
}

/**
 * 分步衰减调度器。
 * $\eta = \frac{\eta_0}{1 + epoch \cdot dropRate}$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class StepDecayScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate / (1 + epoch * dropRate)
}

/**
 * 指数衰减调度器。
 * $\eta = \eta_0 \cdot dropRate^{epoch}$
 * @see [doc/neural.md](../../doc/neural.md)
 */
class ExponentialScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate * (dropRate.pow(epoch))
}