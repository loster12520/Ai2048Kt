package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.plus

infix fun D2Array<Double>.addB(b: D1Array<Double>): D2Array<Double> {
    require(shape[1] == b.shape[0])
    return this + mk.ndarray(b.data.map { value ->
        (0..<shape[0]).map { value }
    }).transpose()
}