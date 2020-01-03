from menpo.visualize import print_dynamic
from menpofit.fittingresult import compute_error
import menpodetect

from . import util


def test_model(model, test_images, num_init):
    face_detector = menpodetect.load_dlib_frontal_face_detector()
    test_gt_shapes = util.get_gt_shapes(test_images)
    test_boxes = util.get_bounding_boxes(test_images, test_gt_shapes, face_detector)

    initial_errors = []
    final_errors = []

    initial_shapes = []
    final_shapes = []

    for k, (im, gt_shape, box) in enumerate(zip(test_images, test_gt_shapes, test_boxes)):
        init_shapes, fin_shapes = model.apply(im, ([box], num_init, None))

        init_shape = util.get_median_shape(init_shapes)
        final_shape = fin_shapes[0]

        initial_shapes.append(init_shape)
        final_shapes.append(final_shape)

        initial_errors.append(compute_error(init_shape, gt_shape))
        final_errors.append(compute_error(final_shape, gt_shape))

        print_dynamic('{}/{}'.format(k + 1, len(test_images)))

    return initial_errors, final_errors, initial_shapes, final_shapes


def fit_all(model_builder, train_images, test_images, num_init):
    face_detector = menpodetect.load_dlib_frontal_face_detector()

    train_gt_shapes = util.get_gt_shapes(train_images)
    train_boxes = util.get_bounding_boxes(train_images, train_gt_shapes, face_detector)

    model = model_builder.build(train_images, train_gt_shapes, train_boxes)

    initial_errors, final_errors, initial_shapes, final_shapes = test_model(model, test_images, num_init)

    return initial_errors, final_errors, initial_shapes, final_shapes, model
