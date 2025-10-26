package com.lignting.tensor

import com.lignting.number.DoubleNumber
import com.lignting.number.IntNumber
import com.lignting.number.NumberTypes

/**
 * ## n维向量
 *
 * 这是一个n维向量实现，支持基本的形状、数据存储和反向传播功能。
 *
 * 注意，任意的`Tensor`实例均不可变，即其`shape`和`data`属性在实例化后均不可更改。
 *
 * @param shape 形状
 * @param data 数据
 * @param backwardFunction 反向传播函数
 * @param updateList 需要更新的张量列表
 */
class Tensor<T : NumberTypes>(
    val shape: IntArray,
    val data: Array<T>,
    val backwardFunction: (Tensor<T>) -> Unit = { },
    val updateList: List<Tensor<T>> = emptyList(),
)

/**
 * ## 创建一个标量张量
 *
 * 创建一个一维标量张量
 *
 * @param values 数值
 * @return 张量
 */
inline fun <reified T : NumberTypes> tensorOf(vararg values: T): Tensor<T> {
    return Tensor(
        shape = intArrayOf(values.size),
        data = values.toList().toTypedArray(),
    )
}

fun tensorIntOf(vararg values: Int): Tensor<IntNumber> =
    tensorOf(*values.map { IntNumber(it) }.toTypedArray())

fun tensorDoubleOf(vararg values: Double): Tensor<DoubleNumber> =
    tensorOf(*values.map { DoubleNumber(it) }.toTypedArray())

/**
 * ## 重塑张量形状
 *
 * 重塑张量的形状为指定的维度
 *
 * @param values 新的形状
 * @return 重塑后的张量
 */
fun <T : NumberTypes> Tensor<T>.reshape(vararg values: Int): Tensor<T> =
    Tensor(
        shape = values,
        data = this.data,
    )

/**
 * ## 扩展张量维度
 *
 * 将指定维度转变成一个List返回给调用者
 *
 * 例如，`shape`为`[3,2]`的张量`[1,2,3,4,5,6]`，调用`expand(0)`后，变成了`[[1,2,3],[4,5,6]]`；调用`expand(1)`后，变成了`[[1,4],[2,5],[3,6]]`
 *
 * 注意：该操作不会改变张量的实际数据存储，仅仅是改变了数据的视图
 *
 * @param dimension 维度索引
 * @return `Tensor`列表
 */
inline fun <reified T : NumberTypes> Tensor<T>.expand(dimension: Int): List<Tensor<T>> {
    require(dimension in shape.indices) { "dimension $dimension out of range for shape ${shape.contentToString()}" }
    val size = shape[dimension]
    val resultShape = shape.filterIndexed { idx, _ -> idx != dimension }.toIntArray()
    val result = List(size) { mutableListOf<T>() }
    val strides = IntArray(shape.size) { i ->
        shape.drop(i + 1).fold(1) { acc, v -> acc * v }
    }
    for (idx in data.indices) {
        var remain = idx
        val indices = IntArray(shape.size)
        for (i in shape.indices) {
            indices[i] = remain / strides[i]
            remain %= strides[i]
        }
        result[indices[dimension]].add(data[idx])
    }
    return result.map { Tensor(resultShape, it.toTypedArray()) }
}

/**
 * ## 广播Tensor
 *
 * 将指定维度的长度从1扩展到新的长度
 *
 * 例如，`shape`为`[3,1,2]`的张量`[[[1,2,3]],[[4,5,6]]]`，调用`broadcastOne(1,2)`后，变成了`[[[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]]]`，shape为`[3,2,2]`
 *
 * 注意：该操作会返回一个新的Tensor，实际数据存储不会被改变
 *
 * @param dim 维度索引
 * @param newLength 新的长度
 * @return 广播后的Tensor
 */
inline fun <reified T : NumberTypes> Tensor<T>.broadcastOne(dim: Int, newLength: Int): Tensor<T> =
    if (dim in shape.indices) {
        broadcastOneStrict(this, dim, newLength)
    } else {
        val newShape = List(dim + 1) { shape.getOrNull(it) ?: 1 }.toIntArray()
        broadcastOneStrict(this.reshape(*newShape), dim, newLength)
    }

/**
 * ## 广播Tensor到新长度
 *
 * 将指定维度的长度从1扩展到新的长度
 *
 * 例如，`shape`为`[3,1,2]`的张量`[[[1,2,3]],[[4,5,6]]]`，调用`broadcastOne(tensor, 1, 2)`后，变成了`[[[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]]]`，shape为`[3,2,2]`
 *
 * 注意：该操作会返回一个新的Tensor，实际数据存储不会被改变；以及，这个函数默认传入的维度一定在当前Tensor的范围内
 *
 * @param tensor 需要广播的Tensor
 * @param dim 维度索引
 * @param newLength 新的长度
 * @return 广播后的Tensor
 */
inline fun <reified T : NumberTypes> broadcastOneStrict(tensor: Tensor<T>, dim: Int, newLength: Int): Tensor<T> {
    assert(tensor.shape[dim] == 1) { "Can only broadcast dimension of size 1, but got ${tensor.shape[dim]} at dimension $dim" }
    val groupSize = tensor.shape.filterIndexed { index, _ -> index < dim }.fold(1) { acc, i -> acc * i }
    val newShape = tensor.shape.toMutableList().apply { this[dim] = newLength }.toIntArray()
    val newData = tensor.data.mapIndexed { index, t -> index to t }.groupBy { it.first / groupSize }.map { (_, v) ->
        List(newLength) { v.map { it.second } }.flatten()
    }.flatten().toTypedArray<T>()
    return Tensor(
        shape = newShape,
        data = newData,
        backwardFunction = tensor.backwardFunction,
        updateList = tensor.updateList
    )
}

/**
 * ## 广播Tensor到新形状
 *
 * 将当前Tensor广播到指定的新形状
 *
 * 例如，`shape`为`[3,1,2]`的张量`[[[1,2,3]],[[4,5,6]]]`，调用`broadcast(intArrayOf(3,2,2))`后，变成了`[[[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]]]`，shape为`[3,2,2]`
 *
 * 注意：该操作会返回一个新的Tensor，实际数据存储不会被改变
 *
 * @param newShape 目标形状
 * @return 广播后的Tensor
 */
inline fun <reified T : NumberTypes> Tensor<T>.broadcast(newShape: IntArray): Tensor<T> {
    assert(shape.size <= newShape.size) { "Cannot broadcast to a smaller number of dimensions" }
    val extendShape = IntArray(newShape.size) { index ->
        shape.getOrElse(index) {
            1
        }
    }
    extendShape.zip(newShape).forEachIndexed { index, (oldDim, newDim) ->
        assert(oldDim == newDim || oldDim == 1) {
            "Cannot broadcast dimension $oldDim to $newDim at index $index"
        }
    }
    var result = this
    extendShape.zip(newShape).forEachIndexed { index, (oldDim, newDim) ->
        result = if (oldDim == 1 && newDim > 1) {
            result.broadcastOne(index, newDim)
        } else {
            result
        }
    }
    return result
}

/**
 * ## 判断是否为整数Tensor
 *
 * 判断当前Tensor的数据类型是否为整数类型
 *
 * @return 如果是整数Tensor则返回true，否则返回false
 */
inline fun <reified T : NumberTypes> Tensor<T>.isIntTensor(): Boolean =
    T::class == IntNumber::class

/**
 * ## 判断是否为浮点Tensor
 *
 * 判断当前Tensor的数据类型是否为浮点类型
 *
 * @return 如果是浮点Tensor则返回true，否则返回false
 */
inline fun <reified T : NumberTypes> Tensor<T>.isDoubleTensor(): Boolean =
    T::class == DoubleNumber::class

/**
 * ## 转换为整数Tensor
 *
 * 将当前Tensor转换为整数类型的Tensor
 *
 * 注意：调用本函数不会继承原有Tensor的反向传播函数和更新列表
 *
 * @return 转换后的整数Tensor
 */
inline fun <reified T : NumberTypes> Tensor<T>.toIntTensor(): Tensor<IntNumber> =
    if (T::class == IntNumber::class) {
        this as Tensor<IntNumber>
    } else {
        Tensor(
            shape = this.shape,
            data = this.data.map {
                when (it) {
                    is IntNumber -> it
                    is DoubleNumber -> IntNumber((it as Double).toInt())
                    else -> throw IllegalArgumentException("Unsupported type")
                }
            }.toTypedArray()
        )
    }

/**
 * ## 转换为浮点Tensor
 *
 * 将当前Tensor转换为浮点类型的Tensor
 *
 * 注意：调用本函数不会继承原有Tensor的反向传播函数和更新列表
 *
 * @return 转换后的浮点Tensor
 */
inline fun <reified T : NumberTypes> Tensor<T>.toDoubleTensor(): Tensor<DoubleNumber> =
    if (T::class == DoubleNumber::class) {
        this as Tensor<DoubleNumber>
    } else {
        Tensor(
            shape = this.shape,
            data = this.data.map {
                when (it) {
                    is DoubleNumber -> it
                    is IntNumber -> DoubleNumber((it as Int).toDouble())
                    else -> throw IllegalArgumentException("Unsupported type")
                }
            }.toTypedArray()
        )
    }