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

# Train the model
function train(model, x_train, y_train, x_val, y_val; epochs=10, lr=0.01)
    opt = Flux.ADAM(params(model), lr=lr)

    for epoch in 1:epochs
        Flux.train!(loss, model, x_train, y_train, opt)

        train_loss = mean(loss(model, x_train, y_train))
        train_acc = accuracy(model, x_train, y_train)

        val_loss = mean(loss(model, x_val, y_val))
        val_acc = accuracy(model, x_val, y_val)

        println("Epoch: $(epoch)")
        println("  Train Loss: $(train_loss)")
        println("  Train Acc: $(train_acc)")
        println("  Val Loss: $(val_loss)")
        println("  Val Acc: $(val_acc)")
    end
end
