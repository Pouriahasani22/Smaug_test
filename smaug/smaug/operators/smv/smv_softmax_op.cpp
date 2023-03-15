#include "smaug/operators/smv/smv_softmax_op.h"
#include "smaug/operators/smv/smv_kernels.h"
#include "smaug/utility/debug_stream.h"
#include <fstream>
namespace smaug {

void SmvSoftmaxOp::tile() {
    auto inputs = getInput(0);
    auto outputs = getOutput(0);
    const TensorShape& shape = inputs->getShape();
    if (shape.getStorageDim(1) >
        SmvBackend::SpadSize() / inputs->getDataTypeSize()) {
        assert(false && "For softmax, a single tile must fit in the local "
                        "scratchpad size!");
    }
    // We can only tile on the N dimension.
    int maxInputs =
            std::min(SmvBackend::SpadSize() / inputs->getDataTypeSize() /
                             shape.getStorageDim(1),
                     shape[0]);
    TensorShape tileShape(
            { maxInputs, shape[1] }, DataLayout::NC, SmvBackend::Alignment);
    tiledTensors[0] = generateTiledTensor(inputs, tileShape, this);
    tiledTensors[1] = generateTiledTensor(outputs, tileShape, this);
}

void SmvSoftmaxOp::run() {
    TiledTensor& inputs = tiledTensors[0];
    TiledTensor& outputs = tiledTensors[1];
    assert(inputs.size() == outputs.size());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs", getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_results", getOutputsMemType());



    for (int i = 0; i < inputs.size(); i++) {
        dout(1) << "Input: " << i << ", output: " << i << "\n";
        Tensor* inputTile = inputs.getTileWithData(i);
        Tensor* outputTile = outputs[i];
        const TensorShape& inputShape = inputTile->getShape();
        const TensorShape& outputShape = outputTile->getShape();
        std::cout <<"\n*Softmax"<<std::endl;
//        std::ofstream summary_file;
//        summary_file.open("./outputs/nnet_fwd_summary_5",std::ios::app);
//        if(!summary_file.fail()) {
//
////        summary_file << "\n******Accelerator Id: " <<smv::kEltwiseOpHw<<"*****"<<std::endl;
//        summary_file << "inputTileIdx: " <<i<<",\t";
//        summary_file << "outputTileIdx: " <<i<<std::endl;
//#if 1
//        summary_file<<"input Tile dimension: ";
//        for(int j{0};j<inputShape.ndims();j++){
////            if(inputShape !=NULL)
//                summary_file << "[" << j << "]: " << inputShape[j]<<"\t";
//        }
//        summary_file <<"\noutput tile dimension: ";
//        for(int j{0};j<outputShape.ndims();j++){
////            if(outputShape !=NULL)
//                summary_file << "[" << j << "]: " << outputShape[j]<<"\t";
//        }
//        summary_file << std::endl;
//#endif
//        }
//        else
//            std::cout << "failed to write to summary file"<<std::endl;
//        summary_file.close();
        mapArrayToAccel(smv::kEltwiseOpHw, "host_inputs",
                        inputTile->data<float16>(),
                        inputShape.storageSize() * sizeof(float16));
        mapArrayToAccel(smv::kEltwiseOpHw, "host_results",
                        outputTile->data<float16>(),
                        outputShape.storageSize() * sizeof(float16));
        invokeKernel(smv::kEltwiseOpHw, smv_softmax_nc_vec_fxp,
                     inputTile->data<float16>(), outputTile->data<float16>(),
                     smv::spad0, smv::spad1, inputShape[0], inputShape[1],
                     inputShape.getPadding(1));
    }
    {
        auto stats = gem5::ScopedStats(
                stats::kTensorFinalStart, stats::kTensorFinalEnd);
        outputs.untile();
    }
}

}  // namespace smaug
