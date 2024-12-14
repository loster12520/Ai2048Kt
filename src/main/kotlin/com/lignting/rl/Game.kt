package com.lignting.rl

import kotlin.math.abs
import kotlin.math.pow
import kotlin.random.Random

class Game(
    private val size: Int = 4,
    private val odds: Double = 0.9,
    seed: Long = System.currentTimeMillis(),
) {
    var panel = MutableList(size) { MutableList(size) { 0 } }

    private val random = Random(seed)

    private var lastPanel = panel

    private var step = 0

    init {
        newBlock()
    }

    private fun newBlock() = panel.mapIndexed { x, col -> col.mapIndexed { y, item -> (x to y) to item } }.flatten()
        .filter { it.second == 0 }.random().also {
            panel[it.first.first][it.first.second] =
                if (random.nextDouble() < odds) 1 else 2
        }.let { this }

    fun isContinue() = panel.flatten().any { it == 0 } or
            panel.mapIndexed { x, col -> List(col.size) { y -> x to y } }.flatten().any {
                if (it.first == 0) false else panel[it.first][it.second] == panel[it.first - 1][it.second] ||
                        if (it.first == size - 1) false else panel[it.first][it.second] == panel[it.first + 1][it.second] ||
                                if (it.second == 0) false else panel[it.first][it.second] == panel[it.first][it.second - 1] ||
                                        if (it.second == size - 1) false else panel[it.first][it.second] == panel[it.first][it.second + 1]
            }

    fun move(direction: Int): Game {
        lastPanel = panel.map { it.toMutableList() }.toMutableList()
        when (direction) {
            0 -> direction(column = false, line = false)// left
            1 -> direction(column = false, line = true) // right
            2 -> direction(column = true, line = false) // up
            3 -> direction(column = true, line = true)  // down
            else -> throw RuntimeException()
        }
        if (lastPanel.zip(panel).all { it.first.zip(it.second).all { it.first == it.second } }) {
            step++
            if (isContinue())
                if (panel.flatten().any { it == 0 })
                    newBlock()
        }

//        if (step > 10000)
//            println("~~~~success!!")
        return this
    }

    private fun direction(column: Boolean, line: Boolean) =
        panel
            .let { if (column) it.t() else it }
            .map { if (line) it.reversed() else it }
            .map {
                val list = mutableListOf<Int>()
                it.forEach {
                    if (it == 0)
                        return@forEach
                    else if (list.size == 0)
                        list += it
                    else if (it == list[list.size - 1])
                        list[list.size - 1] += 1
                    else
                        list += it
                }
                list + List(size - list.size) { 0 }
            }
            .map { if (line) it.reversed() else it }
            .let { if (column) it.t() else it }
            .let { panel = it.map { it.toMutableList() }.toMutableList() }

    private fun List<List<Int>>.t() =
        indices.map { x ->
            indices.map { y ->
                this[y][x]
            }
        }

    fun print() = panel.forEach {
        println(
            it
//            .map { 2.0.pow(it).toInt() }
        )
    }.let {
        println("score: ${score()}")
        println("step: ${step}")
        println("--------------")
        this
    }

    fun panel() = panel.flatten()

    fun score(): Double {
        return (
                panel.flatten().reduce { acc, list -> acc + list }
                )
            .toDouble()
    }

    @Suppress("NAME_SHADOWING")
    fun reward(): Double {
        val maxValue = panel.mapIndexed { x, it -> it.mapIndexed { y, value -> (x to y) to value } }
            .flatten().maxBy { it.second }
        return panel.mapIndexed { x, col -> col.mapIndexed { y, value -> value - abs(x - maxValue.first.first) - abs(y - maxValue.first.second) } }
            .zip(panel)
            .map {
                it.first.zip(it.second).map {
                    (
                            if (it.first == it.second) 1
                            else
                                (it.second / (it.first - it.second))
                            ).toDouble()
                }
            }.flatten().reduce { acc, list -> acc + list } + step * 10 + score()
    }

    fun step() = step
}


fun main() {
    val game = Game()
        .print().move(0).print().move(0).print().move(0).print().move(0).print()
    println(game.reward())
}