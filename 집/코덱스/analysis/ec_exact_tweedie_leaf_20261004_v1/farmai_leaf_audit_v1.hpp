#ifndef FARMAI_EXACT_LEAF_AUDIT_HPP_
#define FARMAI_EXACT_LEAF_AUDIT_HPP_
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <vector>
namespace LightGBM {
inline void FarmaiAudit(const std::string& line) {
  const char* path=std::getenv("FARMAI_LGB_AUDIT_PATH");
  if (path==nullptr || path[0]=='\0') return;
  static std::mutex lock;
  std::lock_guard<std::mutex> held(lock);
  std::ofstream out(path,std::ios::app);
  if (!out) Log::Fatal("FARMAI audit open failed");
  out<<line<<"\n";
  if (!out) Log::Fatal("FARMAI audit write failed");
}
inline void FarmaiBagAudit(int iteration,const data_size_t* ids,data_size_t count) {
  if (std::getenv("FARMAI_LGB_AUDIT_PATH")==nullptr) return;
  std::ostringstream out;out<<"{\"kind\":\"bag\",\"iteration\":"<<iteration<<",\"ids\":[";
  for(data_size_t i=0;i<count;++i){if(i)out<<",";out<<ids[i];}
  out<<"]}";FarmaiAudit(out.str());
}
}
#endif
