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
            Sigmoid(),
            loss = Mse(),
            learningRate = 0.0001
        )
    ).start()
}