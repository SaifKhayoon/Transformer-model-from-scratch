using Flux, Statistics

# Define the loss function
function loss(model, x, y)
    y_pred = model(x)
    return Flux.crossentropy(y_pred, y)
end

# Define the accuracy metric
function accuracy(model, x, y)
    y_pred = model(x)
    y_pred = onecold(y_pred)
    y = onecold(y)
    return mean(y_pred .== y)
end

# Evaluate the model on the test set
function evaluate(model, x_test, y_test)
    test_loss = mean(loss(model, x_test, y_test))
    test_acc = accuracy(model, x_test, y_test)

    println("Test Loss: $(test_loss)")
    println("Test Acc: $(test_acc)")
end
