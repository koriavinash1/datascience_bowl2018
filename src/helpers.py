import tensorflow as tf

def dice_score(output, target, loss_type='jaccard', axis=[1,2,3], smooth=1e-5):
	inse = tf.reduce_sum(output * target, axis=axis)
	if loss_type == 'jaccard':
		l = tf.reduce_sum(output * output, axis=axis)
		r = tf.reduce_sum(target * target, axis=axis)
	elif loss_type == 'sorensen':
		l = tf.reduce_sum(output, axis=axis)
		r = tf.reduce_sum(target, axis=axis)
	else:
		raise Exception("Unknow loss_type")

	dice = (2. * inse + smooth) / (l + r + smooth)
	##
	dice = tf.reduce_mean(dice)
	return dice