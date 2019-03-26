from ops import *
from utils import *
from data_utils.data_utils import *
from models.network import *
from tqdm import tqdm
import tensorflow as tf
import time

class VPAD():
    def __init__(self, args, inlier_cls):
        self.self = self
        self.args = args

    def define_dataset(self):
        dm_train = data_manager(self.args.dataset, is_training=True)
        self.images_train, self.labels_train = dm_train()
        print_saver("dataset: " + str(self.args.dataset), self.args.save_dir)
        print_saver("Train images: " + str(np.shape(self.images_train)), self.args.save_dir)

        self.bm_train = batch_manager(self.images_train, self.labels_train, self.args.batch_size)

        self.h = np.shape(self.images_train)[1]
        self.w = np.shape(self.images_train)[2]
        self.c = np.shape(self.images_train)[3]

    def define_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[self.args.batch_size, self.h, self.w, self.c], name="X")
        self.z = tf.placeholder(tf.float32, shape = [self.args.batch_size, 128], name = "z")
        self.lr = tf.placeholder(tf.float32, name="lr")
        self.phase = tf.placeholder(tf.bool, name="phase")

    def define_network(self):
        if self.args.dataset == "cifar10":
            self.Discriminator = network("D_Res_cifar10", self.args)
            self.Generator = network("G_Res_cifar10", self.args)
        else:
            raise NotImplementedError

    def define_flow(self):
        self.X_gen = self.Generator(self.z, self.phase, update_collection = None)

        self.dis_fake_logits = self.Discriminator(self.X_gen, self.phase ,update_collection = None)
        self.dis_real_logits = self.Discriminator(self.X, self.phase, update_collection = "NO_OPS")

    def define_loss(self):
        if self.args.loss == "hinge":
            self.D_advsr_loss = tf.reduce_mean(tf.nn.relu(1.0 - self.dis_real_logits)) + tf.reduce_mean(tf.nn.relu(1.0 + self.dis_fake_logits))
            self.G_advsr_loss = - tf.reduce_mean(self.dis_fake_logits)
            self.dist = self.G_advsr_loss
        elif self.args.loss == 'wasserstein':
            self.D_advsr_loss = tf.reduce_mean(tf.nn.softplus(self.dis_fake_logits) + tf.nn.softplus(- self.dis_real_logits))
            self.G_advsr_loss = tf.reduce_mean(tf.nn.softplus(- self.dis_fake_logits))
            self.dist = tf.reduce_mean(self.dis_real_logits - self.dis_fake_logits)
        else:
            raise NotImplementedError

        self.D_loss = self.D_advsr_loss
        self.G_loss = self.G_advsr_loss

    def define_optim(self):
        total_vars = tf.trainable_variables()

        d_vars = [var for var in total_vars if "D_" in var.name]
        d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.args.beta1, beta2=self.args.beta2)

        self.global_step = tf.train.get_or_create_global_step()
        g_vars = [var for var in total_vars if "G_" in var.name]
        g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.args.beta1, beta2=self.args.beta2)

        d_gvs = d_optimizer.compute_gradients(self.D_loss, var_list = d_vars)
        g_gvs = g_optimizer.compute_gradients(self.G_loss, var_list = g_vars)
        self.d_solver = d_optimizer.apply_gradients(d_gvs)
        self.g_solver = g_optimizer.apply_gradients(g_gvs, global_step = self.global_step)

    def define_summary_session(self):
        if not tf.gfile.Exists(self.args.save_dir):
            tf.gfile.MakeDirs(self.args.save_dir)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver()

    def define_dict(self):
        self.fetch_dict_d = {
            "opt": self.d_solver,
            "D_loss": self.D_loss,
            "dist": self.dist
        }

        self.fetch_dict_g = {
            "opt": self.g_solver,
            "step": self.global_step,
            "G_loss": self.G_loss,
            "X_gen": self.X_gen
        }

        self.feed_dict = {
            self.X: None,
            self.z: None,
            self.lr: None,
            self.phase: None
        }

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        print_saver("start train...", self.args.save_dir)
        start_time = time.time()
        for i in tqdm(range(self.args.training_step*1000)):
            train_xs, train_ys = self.bm_train.get_next_batch()
            z_sample = np.random.normal(loc=0, scale=1.0, size=[self.args.batch_size, self.args.n_z])
            self.feed_dict[self.X] = train_xs
            self.feed_dict[self.z] = z_sample
            self.feed_dict[self.phase] = True
            self.feed_dict[self.lr] = self.args.lr
            for k in range(5):
                fetch_dict_d = self.sess.run(self.fetch_dict_d, feed_dict=self.feed_dict)
            fetch_dict_g = self.sess.run(self.fetch_dict_g, feed_dict=self.feed_dict)

            if i % 1000 == 0 or i == (self.args.training_step*1000 -1):
                ckpt_path = os.path.join(self.args.save_dir, "ckpt_" + str(i))
                ckpt = self.saver.save(self.sess, ckpt_path)
                print_saver("SSADGAN`s ckpt files is saved as : " + str(ckpt), self.args.save_dir)
                plot_images(fetch_dict_g["X_gen"][:64], 8, 8, self.args.save_dir + "/plot", str(i))
                print_saver("Step: %d    lr: %f" % (fetch_dict_g["step"], self.args.lr), self.args.save_dir)
                print_saver("D_loss: %f    G_loss: %f" % (fetch_dict_d["D_loss"], fetch_dict_g["G_loss"]), self.args.save_dir)
                if self.args.loss == 'wasserstein':
                    print_saver("W_distance: %f" % (fetch_dict_d["dist"]),self.args.save_dir)
                print_time(start_time, self.args.save_dir)
                print_saver("-" * 80, self.args.save_dir)

        self.sess.close()

    def test(self):
        print_saver("start test...", self.args.save_dir)
        if not tf.gfile.Exists(self.args.save_dir):
            tf.gfile.MakeDirs(self.args.save_dir)

        recent_ckpt_job_path = tf.train.latest_checkpoint(self.args.save_dir)
        print_saver("SSADGAN`s recent ckpt files was saved as : " + str(recent_ckpt_job_path), self.args.save_dir)
        ckpt_restore = self.saver.restore(self.sess, recent_ckpt_job_path)
        print_saver("restore complete!", self.args.save_dir)
        self.IS_FID()

    def IS_FID(self):
        num_images_to_eval = 50000
        eval_images = []
        num_batches = (num_images_to_eval // self.args.batch_size) + 1
        print_saver("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval), self.args.save_dir)
        np.random.seed(0)
        for _ in range(num_batches):
            z_sample = np.random.normal(loc=0, scale=1.0, size=[self.args.batch_size, self.args.n_z])
            images = self.sess.run(self.X_gen, feed_dict={self.z: z_sample, self.phase: False})
            eval_images.append(images)
        np.random.seed()
        eval_images = np.vstack(eval_images)
        eval_images = eval_images[:num_images_to_eval]
        eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        self.images_train = np.clip((self.images_train + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)

        tf.reset_default_graph()
        from evaluation.IS import get_inception_score
        inception_score_mean, inception_score_std = get_inception_score(eval_images)
        print_saver("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std), self.args.save_dir)

        tf.reset_default_graph()
        from evaluation.FID import get_fid
        fid = get_fid(self.images_train, eval_images)
        print_saver("FID Score: %f" % (fid), self.args.save_dir)

