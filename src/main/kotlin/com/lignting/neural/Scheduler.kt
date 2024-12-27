package com.lignting.neural

interface Scheduler {
    fun getLearningRate(epoch: Int): Double
}

class StepDecay(private val basicLearningRate: Double, private val dropRate: Double = 0.99) : Scheduler {
    override fun getLearningRate(epoch: Int) = basicLearningRate / (1 + epoch * dropRate)
}