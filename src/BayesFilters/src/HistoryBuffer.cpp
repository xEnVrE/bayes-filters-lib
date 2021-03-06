#include <BayesFilters/HistoryBuffer.h>

using namespace bfl;
using namespace Eigen;


HistoryBuffer::HistoryBuffer(HistoryBuffer&& history_buffer) noexcept :
    window_(history_buffer.window_),
    history_buffer_(std::move(history_buffer.history_buffer_))
{
    history_buffer.window_ = 0;
}


HistoryBuffer& HistoryBuffer::operator=(HistoryBuffer&& history_buffer) noexcept
{
    if (this != &history_buffer)
    {
        window_ = history_buffer.window_;
        history_buffer.window_ = 0;

        history_buffer_ = std::move(history_buffer.history_buffer_);
    }

    return *this;
}


void HistoryBuffer::addElement(const Ref<const VectorXd>& element)
{
    history_buffer_.push_front(element);

    if (history_buffer_.size() > window_)
        history_buffer_.pop_back();
}


MatrixXd HistoryBuffer::getHistoryBuffer() const
{
    MatrixXd hist_out(7, history_buffer_.size());

    unsigned int i = 0;
    for (const Ref<const VectorXd>& element : history_buffer_)
        hist_out.col(i++) = element;

    return hist_out;
}


bool HistoryBuffer::setHistorySize(const unsigned int window)
{
    unsigned int tmp;
    if      (window == window_)     return true;
    else if (window < 2)            tmp = 2;
    else if (window >= max_window_) tmp = max_window_;
    else                            tmp = window;

    if (tmp < window_ && tmp < history_buffer_.size())
    {
        for (unsigned int i = 0; i < (window_ - tmp); ++i)
            history_buffer_.pop_back();
    }

    window_ = tmp;

    return true;
}


bool HistoryBuffer::decreaseHistorySize()
{
    return setHistorySize(window_ - 1);
}


bool HistoryBuffer::increaseHistorySize()
{
    return setHistorySize(window_ + 1);
}


bool HistoryBuffer::clear()
{
    history_buffer_.clear();
    return true;
}
