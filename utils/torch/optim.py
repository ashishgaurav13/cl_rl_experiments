def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, mode = 0):
    """Decreases the learning rate linearly"""
    assert(mode in [0, 1])
    if mode == 0:
        lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    elif mode == 1:
        final_lr = 0.0
        threshold_lr = 0.5 * initial_lr
        difference = final_lr - initial_lr
        lr = initial_lr + difference * (epoch / float(total_num_epochs))
        lr = max(lr, threshold_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr