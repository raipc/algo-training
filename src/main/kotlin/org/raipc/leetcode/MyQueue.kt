package org.raipc.leetcode
import java.util.*

class MyQueue() {
    private val pushStack = ArrayDeque<Int>()
    private val popStack = ArrayDeque<Int>()

    fun push(x: Int) = pushStack.addLast(x)

    fun pop(): Int {
        if (popStack.isEmpty()) { while (pushStack.isNotEmpty()) popStack.addLast(pushStack.pollLast()) }
        return popStack.removeLast()
    }

    fun peek(): Int {
        if (popStack.isEmpty()) { while (pushStack.isNotEmpty()) popStack.addLast(pushStack.pollLast()) }
        return popStack.peekLast()
    }

    fun empty(): Boolean = pushStack.isEmpty() && popStack.isEmpty()

}