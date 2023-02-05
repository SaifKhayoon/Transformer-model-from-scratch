# colab stuff
using IJulia
notebook(detached=true)

# for local use AMD : ENV["KNET_GPU"] = "vulkan"
using Knet

# Define the architecture
struct Transformer
    # Layer parameters
    Wq, Wk, Wv, Wfc1, Wfc2
    bq, bk, bv, bfc1, bfc2
end

# Initialize the model parameters
function Transformer(input_size::Int, hidden_size::Int, num_heads::Int)
    Wq = param(hidden_size, input_size)
    Wk = param(hidden_size, input_size)
    Wv = param(hidden_size, input_size)
    Wfc1 = param(hidden_size, hidden_size)
    Wfc2 = param(hidden_size, hidden_size)
    bq = param0(hidden_size)
    bk = param0(hidden_size)
    bv = param0(hidden_size)
    bfc1 = param0(hidden_size)
    bfc2 = param0(hidden_size)
    Transformer(Wq, Wk, Wv, Wfc1, Wfc2, bq, bk, bv, bfc1, bfc2)
end

# Define the self-attention function
function self_attention(query, key, value, mask)
    attention_weights = softmax(query * transpose(key) / sqrt(hidden_size))
    masked_attention_weights = attention_weights .* mask
    weighted_value = masked_attention_weights * value
    return weighted_value
end

# Define the feedforward function
function feedforward(x, Wfc1, bfc1, Wfc2, bfc2)
    h = relu(x * Wfc1 .+ bfc1)
    y = h * Wfc2 .+ bfc2
    return y
end

# Define the forward pass function
function (model::Transformer)(input)
    query = input * model.Wq .+ model.bq
    key = input * model.Wk .+ model.bk
    value = input * model.Wv .+ model.bv
    weighted_value = self_attention(query, key, value, mask)
    y = feedforward(weighted_value, model.Wfc1, model.bfc1, model.Wfc2, model.bfc2)
    return y
end

# Training the model
function train(model, data, labels)
    optimizer = optimizers(model, Adagrad)
    for (x, y) in zip(data, labels)
        # Compute the loss
        y_pred = model(x)
        loss = mean(abs2, y_pred - y)

        # Update the model parameters
        grads = grad(loss, params(model))
        update!(optimizer, grads)
    end
end

# Evaluating the model
function evaluate(model, data, labels)
    accuracy = 0
    for (x, y) in zip(data, labels)
        y_pred = model(x)
        accuracy += mean(y_pred .== y)
    end
    accuracy /= length(data)
    return accuracy
end
