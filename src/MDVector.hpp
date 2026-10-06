/*
    To be improved:
    Add shape getter
    Think more about allowing shape to have zeros in it
    Add reshaping/resizing
    Add slicing (maybe using std::ranges??? I need to check if that even makes sense)
*/

#pragma once
#include <concepts>
#include <cmath>
#include <stdexcept>
#include <vector>

// Helper concepts
namespace {
    template<typename T>
    concept Addable = requires (T a, T b) { a + b; a += b; };
    template<typename T>
    concept Subtractable = requires (T a, T b) { a - b; a -= b; };
    template<typename T>
    concept Multiplicable = requires (T a, T b) { a * b; a *= b; };
    template<typename T>
    concept Divisble = requires (T a, T b) { a / b; a /= b; };
}

template<typename T>
struct MDVector {
    std::vector<T> data;
    std::vector<std::size_t> shape; //numpy style grid shape
    std::vector<std::size_t> steps;
    
    void calcSteps();

    template<typename ...Idx>
    requires (std::is_convertible_v<Idx, std::size_t> && ...)
    std::size_t calcIndex(Idx ...idx) const;

    public:
    MDVector() = default;
    // Constructs MDVector with given shape with default initialized values.
    MDVector(std::vector<std::size_t>) requires std::default_initializable<T>;
    // Constructs MDVector with given shape filled with given value.
    MDVector(std::vector<std::size_t>, const T&);
    // Constructs MDVector with given shape and given underlying array.
    MDVector(std::vector<std::size_t>, T*, T*);

    template<typename ...Idx>
    requires (std::is_convertible_v<Idx, std::size_t> && ...)
    T &operator[](Idx ...idx);

    template<typename ...Idx>
    requires (std::is_convertible_v<Idx, std::size_t> && ...)
    const T &operator[](Idx ...idx) const;

    MDVector<T>& operator+=(MDVector<T>) requires Addable<T>; //Elementwise addition.
    MDVector<T>& operator-=(MDVector<T>) requires Subtractable<T>; //Elementwise subtraction.
    MDVector<T>& operator*=(T) requires Multiplicable<T>; //Elementwise multiplication.
    MDVector<T>& operator/=(T) requires Divisble<T>;
    friend MDVector<T> operator+(MDVector<T> left, const MDVector<T>& right) requires Addable<T> { //Elementwise addition.
        return left += right;
    };
    friend MDVector<T> operator-(MDVector<T> left, const MDVector<T>& right) requires Subtractable<T> { //Elementwise subtraction.
        return left -= right;
    }
    friend MDVector<T> operator*(MDVector<T> vector, const T& scalar) requires Multiplicable<T> { //Elementwise multiplication.
        return vector *= scalar;
    }
    friend MDVector<T> operator*(const T& scalar, MDVector<T> vector) requires Multiplicable<T> { //Elementwise multiplication.
        return vector *= scalar;
    }
    friend T norm(const MDVector<T> &vec) {
        T result = 0;
        for (std::size_t i = 0; i < vec.data.size(); i++) {
            result += vec.data[i]*vec.data[i];
        }
        
        return std::sqrt(result);
    }
};

template<typename T>
void MDVector<T>::calcSteps() {
    steps[steps.size()-1] = 1;
    if (steps.size() == 1) return;
    for (std::size_t i = steps.size()-2; i <= steps.size()-2; i--) {
        steps[i] = steps[i+1] * shape[i+1];
    }
}

template<typename T>
MDVector<T>::MDVector(std::vector<std::size_t> shape) requires std::default_initializable<T> : shape(shape), steps(shape.size()) {
    std::size_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    
    data = std::vector<T>(size);
    calcSteps();
}

template<typename T>
MDVector<T>::MDVector(std::vector<std::size_t> shape, const T &fill_value) : shape(shape), steps(shape.size()) {
    std::size_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    
    data = std::vector<T>(size, fill_value);
    calcSteps();
}

template<typename T>
template<typename... Idx>
requires (std::is_convertible_v<Idx, std::size_t> && ...)
std::size_t MDVector<T>::calcIndex(Idx... idx) const {
    std::vector<std::size_t> indices { static_cast<std::size_t>(idx)... };
    std::size_t index = 0;
    for (std::size_t i = 0; i < steps.size(); i++) {
        index += indices[i] * steps[i];
    }

    return index;
}

template<typename T>
template<typename... Idx>
requires (std::is_convertible_v<Idx, std::size_t> && ...)
T &MDVector<T>::operator[](Idx... idx) {
    return data[calcIndex(idx...)];
}

template<typename T>
template<typename... Idx>
requires (std::is_convertible_v<Idx, std::size_t> && ...)
const T &MDVector<T>::operator[](Idx... idx) const {
    return data[calcIndex(idx...)];
}

template <typename T>
MDVector<T> &MDVector<T>::operator+=(MDVector<T> other) requires Addable<T> {
    if (shape != other.shape) {
        throw std::invalid_argument("Addition of MDVectors of unequal sizes.");
    }
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] += other.data[i];
    }

    return *this;
}

template <typename T>
MDVector<T> &MDVector<T>::operator-=(MDVector<T> other) requires Subtractable<T> {
    if (shape != other.shape) {
        throw std::invalid_argument("Subtraction of MDVectors of unequal sizes.");
    }
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] -= other.data[i];
    }

    return *this;
}

template <typename T>
MDVector<T> &MDVector<T>::operator*=(T scalar) requires Multiplicable<T> {
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] *= scalar;
    }

    return *this;
}

template <typename T>
MDVector<T> &MDVector<T>::operator/=(T scalar) requires Divisble<T> {
	for (std::size_t i = 0; i < data.size(); i++) {
	    data[i] /= scalar;
	}

	return *this;
}
