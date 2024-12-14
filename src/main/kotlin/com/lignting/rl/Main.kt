package com.lignting.rl

import com.lignting.neural.*

fun main() {
    Agent(
        Model(
            Dense(16, 13),
            Relu(),
//            Dropout(),
            Dense(13, 10),
            Relu(),
//            Dropout(),
            Dense(10, 7),
            Relu(),
//            Dropout(),
            Dense(7, 4),
            Relu(),
//            Dropout(),
            loss = Mse(),
            learningRate = 0.0001
        )
    ).start().paintLoss().paintGame()
}