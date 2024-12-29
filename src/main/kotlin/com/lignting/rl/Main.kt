package com.lignting.rl

import com.lignting.neural.*

fun main() {
    Agent(
        Model(
            Dense(16, 1024),
            LeakyRelu(),
            Dropout(),
            Dense(1024, 256),
            LeakyRelu(),
            Dropout(),
            Dense(256, 128),
            LeakyRelu(),
            Dropout(),
            Dense(128, 4),
            LeakyRelu(),
            Dropout(),
            loss = Mae(),
            optimizer = Adam(),
            scheduler = ExponentialScheduler(1e-3, dropRate = 1 - 1e-2)
        )
    ).start().paintLoss().paintGame()
}