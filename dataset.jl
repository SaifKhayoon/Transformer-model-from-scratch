using JuliaDB, CSV

# Load the dataset
function load_dataset(filename)
    data = CSV.File(filename) |> DataFrame!
    return data
end

# Preprocess the dataset
function preprocess_dataset(data)
    # Normalize the input features
    x = convert(Matrix, data[:, 1:end-1])
    x = (x .- mean(x, dims=1)) ./ std(x, dims=1)

    # Convert the labels to one-hot encoding
    y = onehotbatch(data[:, end])

    return x, y
end

# Split the dataset into training and validation sets
function split_dataset(x, y, train_ratio)
    n = size(x, 1)
    n_train = convert(Int, n * train_ratio)

    x_train, y_train = x[1:n_train, :], y[1:n_train, :]
    x_val, y_val = x[n_train+1:end, :], y[n_train+1:end, :]

    return x_train, y_train, x_val, y_val
end
using JuliaDB, CSV

# Load the dataset
function load_dataset(filename)
    data = CSV.File(filename) |> DataFrame!
    return data
end

# Preprocess the dataset
function preprocess_dataset(data)
    # Normalize the input features
    x = convert(Matrix, data[:, 1:end-1])
    x = (x .- mean(x, dims=1)) ./ std(x, dims=1)

    # Convert the labels to one-hot encoding
    y = onehotbatch(data[:, end])

    return x, y
end

# Split the dataset into training and validation sets
function split_dataset(x, y, train_ratio)
    n = size(x, 1)
    n_train = convert(Int, n * train_ratio)

    x_train, y_train = x[1:n_train, :], y[1:n_train, :]
    x_val, y_val = x[n_train+1:end, :], y[n_train+1:end, :]

    return x_train, y_train, x_val, y_val
end
