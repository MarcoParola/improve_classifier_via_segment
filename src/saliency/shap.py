import shap
import numpy as np
import os
import matplotlib.pyplot as plt


class OralShap:

    def create_maps_shap(model, test_loader):

        for blocks in model.model.named_modules():
            for mod in blocks:
                if hasattr(mod, "inplace"):
                    # print(mod)
                    mod.inplace = False
        print("A")

        class_names = ['Neoplastic', 'Aphthous', 'Traumatic']

        os.makedirs('test_shap', exist_ok=True)
        model.eval()


        for batch_index, (images, _) in enumerate(test_loader):
            print("B")
            e = shap.GradientExplainer((model, model.model.features[-1][-1]), images, batch_size=len(images))
            print("C")
            shap_values, indexes = e.shap_values(images, ranked_outputs=1, nsamples=50)
            print("D")
            shap_array = shap_values[0]
            for i in range(len(indexes)):
                # get the class name corresponding to the index
                class_name = class_names[indexes[i]]
                print("1")

                # overlay the saliency map on the image
                image_for_plot = images[i].permute(1, 2, 0).cpu().numpy()
                current_shap_value = shap_array[i]
                print("2")

                shap_image = current_shap_value[0]
                print("3")


                fig, ax = plt.subplots()
                ax.imshow(image_for_plot)

                ax.imshow((shap_image*255).astype('uint8'), cmap='jet', alpha=0.75)  # Overlay saliency map

                # save the figure with the overlaid saliency map
                plt.savefig(os.path.join('test_shap', f'saliency_map_batch_{batch_index}_image_number_{i}_class_{class_name}.png'), bbox_inches='tight')
                plt.close()
                print(shap_image)




