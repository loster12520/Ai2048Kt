package com.lignting.rl

import com.lignting.neural.*

fun main() {
    Agent(
        Model(
            Dense(16, 10),
            Relu(),
            Dense(10, 8),
            Relu(),
            Dense(8, 4),
            Relu(),
            loss = Mae(),
            learningRate = 0.001
        )
    ).start()
}