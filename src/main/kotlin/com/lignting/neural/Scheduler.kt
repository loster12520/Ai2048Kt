package com.lignting.neural

import kotlin.math.pow

interface Scheduler {
    fun getLearningRate(epoch: Int): Double
}

class StepDecayScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate / (1 + epoch * dropRate)
}

class ExponentialScheduler(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate * (dropRate.pow(epoch))
}