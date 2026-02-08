import numpy as np
from river import optim
from river import linear_model
from river.tree import HoeffdingTreeRegressor
import matplotlib.pyplot as plt


class WeightAdjustingBinaryTransformation:
    def __init__(self, n_labels, increase_factor=1.0, decrease_factor=0.1):
        self.n_labels = n_labels
        self.model = self._get_model()

        self.label_weights = [0.0] * self.n_labels
        self.current_order = list(range(self.n_labels))

        self.inc = increase_factor
        self.dec = decrease_factor

        self.streaks = [0] * self.n_labels

        self.upper_weight_limit = float('inf')
        self.lower_weight_limit = float(0)

        # FOR PLOTTING, REMOVABLE
        self.weight_history = []

    def learn_one(self, X, y):
        self._update_weights(y)

        self._update_order()

        y_ordered = self._reorder_y(y)

        C = self._transform(y_ordered)

        self.model.learn_one(X, C)

        # Weight history is kept only for plotting.
        # It might slightly affect performance,
        # so uncomment if you're doing sensitive time/memory efficiency experiments.
        self.weight_history.append(self.label_weights.copy())

    def predict_one(self, X):
        C_pred = np.round(self.model.predict_one(X)).astype(np.int32)

        y_pred_ordered = self._binarize(C_pred)

        y_pred_original = self._reverse_reorder_y(y_pred_ordered)

        return y_pred_original

    def sigmoid(self, x):
        safe_x = max(-20, min(x, 20))

        delay_factor = 3

        return 1.0 + (1.0 / (1.0 + np.exp(-(safe_x-delay_factor))))

    def _update_weights(self, y):
        y_list = list(y.values())
        for index, bit in enumerate(y_list):
            if bit == 1:
                if self.streaks[index] < 0:
                    self.streaks[index] = 0
                self.streaks[index] += 1

                acceleration = self.sigmoid(self.streaks[index])

                self.label_weights[index] = min(self.upper_weight_limit, self.label_weights[index] + self.inc * acceleration)
            else:
                if self.streaks[index] > 0:
                    self.streaks[index] = 0
                self.streaks[index] -= 1

                deacceleration = self.sigmoid(-self.streaks[index])

                self.label_weights[index] = max(self.label_weights[index] - deacceleration * self.dec, self.lower_weight_limit)


    def _update_order(self):
        self.current_order = np.argsort(np.array(self.label_weights), kind='mergesort').tolist()

    def _reorder_y(self, y):
        y_list_original = list(y.values())
        y_ordered = [0] * self.n_labels
        for new_index, original_index in enumerate(self.current_order):
            y_ordered[new_index] = y_list_original[original_index]
        return y_ordered

    def _reverse_reorder_y(self, y_ordered):
        y_original = [0] * self.n_labels
        for new_index, original_index in enumerate(self.current_order):
            y_original[original_index] = y_ordered[new_index]
        return np.array(y_original, dtype=np.int32)

    def _get_model(self):
        return HoeffdingTreeRegressor(
            leaf_model=linear_model.LinearRegression(
                optimizer=optim.Adam(lr=1e-5),
                intercept_lr=0.5,
                loss=optim.losses.Squared()
            )
        )

    def _transform(self, y_list):
        C = 0
        for bit in y_list:
            C = (C << 1) | bit
        return C

    def _binarize(self, C):
        max_val = (2 ** self.n_labels) - 1
        if C > max_val or C < 0:
            return np.ones(self.n_labels, dtype=np.int32)

        bin_str = np.binary_repr(C, width=self.n_labels)
        bin_list = [int(bit) for bit in bin_str]

        return np.array(bin_list, dtype=np.int32)

    def plot_weight_history(self, dataset_title, skip=10, top_n=20):

        if not self.weight_history:
            print("No weight history recorded yet.")
            return

        history = np.array(self.weight_history)
        steps = np.arange(len(history))

        final_weights = history[-1]
        top_indices = np.argsort(final_weights)[-top_n:][::-1]

        percentages = steps / (len(history) - 1) * 100

        plt.figure(figsize=(10, 6))
        for i in top_indices:
            plt.plot(percentages[::skip], history[::skip, i], alpha=0.8, linewidth=1.5, label=f'Label {i}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

        plt.title(f"Top {top_n} Label Weights Over Time - Dataset: {dataset_title}")
        plt.xlabel("Percentage of Instances Covered (%)")
        plt.ylabel("Weight Value")
        plt.xticks(np.arange(0, 110, 10))
        plt.grid(True, linestyle='--', alpha=0.4)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(f"./results/{dataset_title}_weight_history.eps", format='eps')
        plt.show()
