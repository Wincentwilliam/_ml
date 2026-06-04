import java.util.EmptyStackException;

public class StackDemo {
    private int maxSize;
    private int[] stackArray;
    private int top;

    public StackDemo(int size) {
        this.maxSize = size;
        this.stackArray = new int[maxSize];
        this.top = -1;
    }

    public void push(int value) {
        if (top == maxSize - 1) {
            System.out.println("Stack Overflow! Cannot push " + value);
            return;
        }
        stackArray[++top] = value;
        System.out.println("Pushed: " + value);
    }

    public int pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return stackArray[top--];
    }

    public int peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return stackArray[top];
    }

    public boolean isEmpty() {
        return (top == -1);
    }

    public static void main(String[] args) {
        StackDemo myStack = new StackDemo(10);
        int[] numbersToPush = {10, 20, 30, 40, 50};

        System.out.println("--- Pushing 5 numbers ---");
        for (int num : numbersToPush) {
            myStack.push(num);
        }

        System.out.println("\nTop element (peek): " + myStack.peek());

        System.out.println("\n--- Popping elements ---");
        while (!myStack.isEmpty()) {
            System.out.println("Popped: " + myStack.pop());
        }
    }
}