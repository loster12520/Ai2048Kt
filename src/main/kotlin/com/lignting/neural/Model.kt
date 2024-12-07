package com.lignting.neural

class Model(vararg layers: Layer) {
    val layerList = layers.toList()
    fun fit() {}
    fun predict() {}
    fun copy() {}
}