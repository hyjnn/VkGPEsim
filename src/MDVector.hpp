/*
    To be improved:
    Think more about allowing shape to have zeros in it
    Add reshaping/resizing
    Add slicing (maybe using std::ranges??? I need to check if that even makes sense)
*/

#pragma once
#include <concepts>
#include <cmath>
#include <generator>
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
class MDVector {
public:
    constexpr MDVector() = default;
    // Constructs MDVector with given shape with default initialized values.
    constexpr MDVector(std::vector<std::size_t>) requires std::default_initializable<T>;
    // Constructs MDVector with given shape filled with given value.
    constexpr MDVector(std::vector<std::size_t>, const T&);
    // Constructs MDVector with given shape and given underlying array.
    constexpr MDVector(std::vector<std::size_t>, T*, T*);

    template<typename ...Idx>
    requires (std::is_convertible_v<Idx, std::size_t> && ...)
    constexpr T &operator[](Idx ...idx);

    template<typename ...Idx>
    requires (std::is_convertible_v<Idx, std::size_t> && ...)
    constexpr const T &operator[](Idx ...idx) const;

    T &operator[](const std::vector<std::size_t> &MDi);
    const T &operator[](const std::vector<std::size_t> &MDi) const;

    // Iterators, getters etc.

    constexpr auto begin() {
        return data.begin();
    }
    constexpr auto end() {
        return data.end();
    }
    constexpr auto begin() const {
        return data.begin();
    }
    constexpr auto end() const {
        return data.end();
    }

    constexpr T* raw_data() {
        return data.data();
    }
    constexpr const T* raw_data() const {
        return data.data();
    }

    constexpr const std::size_t size() const {
        return data.size();
    }
    constexpr const std::vector<std::size_t> &shape() const {
        return m_shape;
    }

    std::generator<std::vector<std::size_t>> MDIndexGen();

    constexpr MDVector<T>& operator+=(MDVector<T>) requires Addable<T>; //Elementwise addition.
    constexpr MDVector<T>& operator-=(MDVector<T>) requires Subtractable<T>; //Elementwise subtraction.
    constexpr MDVector<T>& operator*=(T) requires Multiplicable<T>; //Elementwise multiplication.
    constexpr MDVector<T>& operator/=(T) requires Divisble<T>;
    friend constexpr MDVector<T> operator+(MDVector<T> left, const MDVector<T>& right) requires Addable<T> { //Elementwise addition.
        return left += right;
    };
    friend constexpr MDVector<T> operator-(MDVector<T> left, const MDVector<T>& right) requires Subtractable<T> { //Elementwise subtraction.
        return left -= right;
    }
    friend constexpr MDVector<T> operator*(MDVector<T> vector, const T& scalar) requires Multiplicable<T> { //Elementwise multiplication.
        return vector *= scalar;
    }
    friend constexpr MDVector<T> operator*(const T& scalar, MDVector<T> vector) requires Multiplicable<T> { //Elementwise multiplication.
        return vector *= scalar;
    }
    friend constexpr T norm(const MDVector<T> &vec) {
        T result = 0;
        for (std::size_t i = 0; i < vec.data.size(); i++) {
            result += vec.data[i]*vec.data[i];
        }
        
        return std::sqrt(result);
    }

private:
    std::vector<T> data;
    std::vector<std::size_t> m_shape; //numpy style grid shape
    std::vector<std::size_t> m_steps;
    
    constexpr void calc_steps();

    constexpr std::size_t calc_index(const std::vector<size_t> &MDi) const;
    template<typename ...Idx>
    requires (std::is_convertible_v<Idx, std::size_t> && ...)
    constexpr std::size_t calc_index(Idx ...idx) const;
};

template<typename T>
constexpr void MDVector<T>::calc_steps() {
    m_steps[m_steps.size()-1] = 1;
    if (m_steps.size() == 1) return;
    for (std::size_t i = m_steps.size()-2; i <= m_steps.size()-2; i--) {
        m_steps[i] = m_steps[i+1] * m_shape[i+1];
    }
}

template<typename T>
constexpr MDVector<T>::MDVector(std::vector<std::size_t> shape) requires std::default_initializable<T> : m_shape(shape), m_steps(shape.size()) {
    std::size_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    
    data = std::vector<T>(size);
    calc_steps();
}

template<typename T>
constexpr MDVector<T>::MDVector(std::vector<std::size_t> shape, const T &fill_value) : m_shape(shape), m_steps(shape.size()) {
    std::size_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    
    data = std::vector<T>(size, fill_value);
    calc_steps();
}


// This function DOES NOT check if the appropriate number of indices was passed
template<typename T>
constexpr std::size_t MDVector<T>::calc_index(const std::vector<std::size_t>& MDi) const {
    std::size_t index = 0;
    for (std::size_t i = 0; i < m_steps.size(); i++) {
        index += MDi[i] * m_steps[i];
    }

    return index;
}

// This function DOES NOT check if the appropriate number of indices was passed
template<typename T>
template<typename... Idx>
requires (std::is_convertible_v<Idx, std::size_t> && ...)
constexpr std::size_t MDVector<T>::calc_index(Idx... idx) const {
    // Since C++20, vectors have constexpr initialization semantics etc.
    std::vector<std::size_t> indices { static_cast<std::size_t>(idx)... };
    return calc_index(indices);
}

template<typename T>
template<typename... Idx>
requires (std::is_convertible_v<Idx, std::size_t> && ...)
constexpr T &MDVector<T>::operator[](Idx... idx) {
    return data[calc_index(idx...)];
}

template<typename T>
template<typename... Idx>
requires (std::is_convertible_v<Idx, std::size_t> && ...)
constexpr const T &MDVector<T>::operator[](Idx... idx) const {
    return data[calc_index(idx...)];
}

template<typename T>
T &MDVector<T>::operator[](const std::vector<std::size_t> &MDi) {
    return data[calc_index(MDi)];
}

template<typename T>
const T &MDVector<T>::operator[](const std::vector<std::size_t> &MDi) const {
    return data[calc_index(MDi)];
}

template <typename T>
constexpr MDVector<T> &MDVector<T>::operator+=(MDVector<T> other) requires Addable<T> {
    if (m_shape != other.m_shape) {
        throw std::invalid_argument("Addition of MDVectors of unequal sizes.");
    }
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] += other.data[i];
    }

    return *this;
}

template <typename T>
constexpr MDVector<T> &MDVector<T>::operator-=(MDVector<T> other) requires Subtractable<T> {
    if (m_shape != other.m_shape) {
        throw std::invalid_argument("Subtraction of MDVectors of unequal sizes.");
    }
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] -= other.data[i];
    }

    return *this;
}

template <typename T>
constexpr MDVector<T> &MDVector<T>::operator*=(T scalar) requires Multiplicable<T> {
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] *= scalar;
    }

    return *this;
}

template <typename T>
constexpr MDVector<T> &MDVector<T>::operator/=(T scalar) requires Divisble<T> {
	for (std::size_t i = 0; i < data.size(); i++) {
	    data[i] /= scalar;
	}

	return *this;
}

template<typename T>
std::generator<std::vector<std::size_t>> MDVector<T>::MDIndexGen() {
    std::vector<std::size_t> MDIndex(m_shape.size(), 0);
    for (std::size_t i = 0; i < size(); i++) {
	co_yield MDIndex;
	for (std::size_t current_dim = m_shape.size(); current_dim--;) {
	    MDIndex[current_dim]++;
	    if (MDIndex[current_dim] < m_shape[current_dim]) break; // No carry to carry
	    MDIndex[current_dim] = 0;
	}
    }
}
