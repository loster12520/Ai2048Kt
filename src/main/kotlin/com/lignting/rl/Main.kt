package com.lignting.rl

import com.lignting.neural.*

fun main() {
    Agent(
        Model(
            Dense(16, 128),
            LeakyRelu(),
            Dropout(),
            Dense(128, 256),
            LeakyRelu(),
            Dropout(),
            Dense(256, 128),
            LeakyRelu(),
            Dropout(),
            Dense(128, 4),
            LeakyRelu(),
            Dropout(),
            loss = Mse(),
            optimizer = Adam(learningRate = 1e-3)
        )
    ).start().paintLoss().paintGame()
}