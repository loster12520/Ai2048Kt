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
            optimizer = Adam(),
            scheduler = ExponentialScheduler(1e-3, dropRate = 1 - 1e-1)
        )
    ).start().paintLoss().paintGame()
}