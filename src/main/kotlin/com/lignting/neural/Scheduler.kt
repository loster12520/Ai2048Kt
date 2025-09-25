package com.lignting.neural

import kotlin.math.pow

/**
 * 学习率调度器接口，定义根据轮次获取学习率的方法。
 */
interface Scheduler {
    /**
     * 获取当前轮次的学习率。
     * @param epoch 当前轮次
     * @return 学习率
     */
    fun getLearningRate(epoch: Int): Double
}

/**
 * 分步衰减学习率调度器。
 * 公式：lr = basicLearningRate / (1 + epoch * dropRate)
 * @param basicLearningRate 初始学习率
 * @param dropRate 衰减因子，默认0.99
 */
class StepDecayScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate / (1 + epoch * dropRate)
}

/**
 * 指数衰减学习率调度器。
 * 公式：lr = basicLearningRate * (dropRate ^ epoch)
 * @param basicLearningRate 初始学习率
 * @param dropRate 衰减因子，默认0.99
 */
class ExponentialScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate * (dropRate.pow(epoch))
}